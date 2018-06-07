import os
import copy
import cv2
import numpy as np
from glob import glob
import tqdm
import math
import os

import torch
from torch.autograd import Variable
from torchvision import models

def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def getImageNetDict():
    path = os.environ.get('IMAGENETDICT_PATH')
    if path is None:
        assert False, "please set IMAGENETDICT_PATH environment variable"
    return eval(open(path).read())

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.cpu().numpy()
    if len(inp.shape) == 4: # with batch dimension
        inp = inp.transpose((0, 2, 3, 1))
    else:
        inp = inp.transpose((1, 2, 0))        
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def convert_np2var(inp):
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    im_as_arr = np.float32(inp)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    if len(im_as_arr.shape) == 4: # with batch dimension
        im_as_arr = im_as_arr.transpose(0, 3, 1, 2)  # Convert array to bs,D,W,H
    else:
        im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H

    # Normalize the channels
    if len(im_as_arr.shape) == 3: # without batch dimension
        for channel, _ in enumerate(im_as_arr):
            #im_as_arr[channel] /= 255 # already in 0-1 range
            im_as_arr[channel] -= mean[channel]
            im_as_arr[channel] /= std[channel]
    else:
        for i in range(im_as_arr.shape[0]):
            for channel, _ in enumerate(im_as_arr[i]):
                #im_as_arr[i][channel] /= 255 # already in 0-1 range
                im_as_arr[i][channel] -= mean[channel]
                im_as_arr[i][channel] /= std[channel]

    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()

    if len(im_as_ten.shape) == 3: # without batch dimension    
        # Add one more channel to the beginning. Tensor shape = 1,3,224,224
        im_as_ten.unsqueeze_(0)

    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def preprocess_im(im, volatile=False):
    original_image = cv2.imread(im)
    prep_img = preprocess_image(original_image)
    if volatile:
        prep_img.volatile = True
    return prep_img.cuda()

def predict(net, im, imagenetdict):
    net.eval()
    prep_img = preprocess_im(im)
    prediction = net(prep_img).data.cpu().numpy().argmax()
    return imagenetdict[prediction], prediction
                
def extract_features(net, images):
    # output nxd features
    net.eval()
    seq = []
    for im in tqdm.tqdm(images):
        prep_img = preprocess_im(im, volatile=True)
        seq.append(prep_img)

    features = []
    for i in tqdm.tqdm(range(0, len(seq), 100)):
        images_batch = torch.cat(seq[i:i+100], dim=0)
        features.append(net(images_batch))
    return torch.cat(features, dim=0)

def load_imagenet_basis(filedir='static/images/tiny-imagenet-200/train/*',
               n_per_class=10):
    # visualize random images in each class of input
    types = glob(filedir)
    res = []
    for t in types:
        all_pics = glob(t + '/images/*')
        # for each 200 category, choose n_per_class
        rs = np.random.choice(range(len(all_pics)), n_per_class, replace=False)
        res.extend([all_pics[r] for r in rs])
    res = np.random.permutation(res)[:4096]
    print('done loading_basis')
    return list(res)

def load_basis(filedir='static/images/color_shape_combined/*'):
    images = np.random.permutation(glob(filedir))[:4096]
    print('done loading_basis')
    return list(images)

def lst_sq_solve(A, Ainv, b):
    # least square solve
    x = Ainv.dot(b)
    SST = ((b-b.mean())**2).sum()
    SSRes = ((b-A.dot(x))**2).sum()
    r_sq = 1 - SSRes / SST
    return x, r_sq
                        
def getContribution(newW, predict_id, 
                    A, Ainv, imagenetdict,
                    basis, net, model,
                    top=10, test_image_name=None, coordinate=True):
    
    if test_image_name is not None:
        word, predict_id = predict(net, test_image_name, imagenetdict)
        print(word, predict_id)
    else:
        word = imagenetdict[predict_id]
        
    # an example explaination: newW 1000 x num_basis
    weights = newW[predict_id]

    x, input_weights = None, None
    if test_image_name is not None:
        b = extract_features(model, [test_image_name])[0].data.cpu().numpy()
        x, r_sq = lst_sq_solve(A, Ainv, b)
        if coordinate:
            criteria = np.abs(x)
        else:
            criteria = np.abs(weights * x)
        args = np.argsort(criteria)[::-1]
    else:
        args = np.argsort(np.abs(weights))[::-1]


    selection = args[:top]
    print('%s =' % word)
    
    theta = weights[selection]
    theta = list(map(float, theta))
    images = np.array(basis)[selection]
    if x is not None:
        input_weights = x[selection]
        input_weights = list(map(float, input_weights))

    return theta, images, input_weights, list(map(int, selection))

def pad_image(im_np, pad=0):
    # im_np is image in numpy format e.g., (bs,w,h,c) or (w, h, c)
    # the final size is (bs, w + 2*p, h + 2*p, c) or (w + 2*p, h + 2*p, c)
    if len(im_np.shape) == 4: #"need batch dimension"
        bs, w, h, c = im_np.shape
        out_im = np.zeros([bs, w + 2*pad, h + 2*pad, c])
        out_im[:, pad:pad+w, pad:pad+h, :] = im_np
    else:
        w, h, c = im_np.shape
        out_im = np.zeros([w + 2*pad, h + 2*pad, c])
        out_im[pad:pad+w, pad:pad+h, :] = im_np        
    return out_im

def slide2d(im_np, kernel_size, pad=0, stride=1, transform=np.sum):
    # im_np is image in numpy format e.g., (bs, w, h, c) or (w, h, c)

    kw, kh = kernel_size
    transform_shape = transform(im_np).shape
    im_np = pad_image(im_np, pad)
    if len(im_np.shape) == 4: # "need batch dimension"
        if len(transform_shape) == 2:
            transform_dim = transform_shape[1]
        else:
            transform_dim = 1
        bs, w, h, c = im_np.shape
        assert w >= kw and h >= kh, "width must be bigger than kernel width"
        new_w, new_h = math.floor((w-kw)/stride) + 1, math.floor((w-kh)/stride) + 1
        out_im = np.zeros([bs, new_w, new_h, transform_dim])
        for i in range(new_w):
            for j in range(new_h):
                out_im[:,i,j,:] = transform(im_np[:,i*stride:(i*stride + kw),
                                                  j*stride:(j*stride + kh),:])
    else:
        if len(transform_shape) == 1:
            transform_dim = transform_shape[0]
        else:
            transform_dim = 1
        
        w, h, c = im_np.shape
        assert w >= kw and h >= kh, "width must be bigger than kernel width"
        new_w, new_h = math.floor((w-kw)/stride) + 1, math.floor((w-kh)/stride) + 1
        out_im = np.zeros([new_w, new_h, transform_dim])
        for i in range(new_w):
            for j in range(new_h):
                out_im[i,j,:] = transform(im_np[i*stride:(i*stride + kw),
                                                j*stride:(j*stride + kh),:])            
    return out_im

def slide2d_tensor(inp, kernel_size, pad=0, stride=1, transform=np.sum):
    # inp is image in pytorch format e.g., (bs, c, w, h) or (c, w, h)

    kw, kh = kernel_size
    transform_shape = transform(inp).shape
    p2d = (pad, pad, pad, pad)
    inp = torch.nn.functional.pad(inp, p2d, 'constant', value=inp.min().data[0])
    if len(inp.shape) == 4: # "need batch dimension"
        if len(transform_shape) == 2:
            transform_dim = transform_shape[1]
        else:
            transform_dim = 1
        bs, c, w, h = inp.shape
        assert w >= kw and h >= kh, "width must be bigger than kernel width"
        new_w, new_h = math.floor((w-kw)/stride) + 1, math.floor((w-kh)/stride) + 1
        out_im = np.zeros([bs, transform_dim, new_w, new_h])
        for i in range(new_w):
            for j in range(new_h):
                out_im[:,:,i,j] = transform(inp[:,:,i*stride:(i*stride + kw),
                                                j*stride:(j*stride + kh)])
    else:
        if len(transform_shape) == 1:
            transform_dim = transform_shape[0]
        else:
            transform_dim = 1
        
        bs, c, w, h = inp.shape
        assert w >= kw and h >= kh, "width must be bigger than kernel width"
        new_w, new_h = math.floor((w-kw)/stride) + 1, math.floor((w-kh)/stride) + 1
        out_im = np.zeros([transform_dim, new_w, new_h])
        for i in range(new_w):
            for j in range(new_h):
                out_im[:,i,j] = transform(inp[:,i*stride:(i*stride + kw),
                                              j*stride:(j*stride + kh)])            
    return out_im

def bilinear_resize(im, OH, OW):
    # im is a pytorch variable or tensor of size (N, C, H, W)
    N, C, H, W = im.shape
    theta = torch.FloatTensor([[1,0,0], [0,1,0]]).unsqueeze(0).expand(N,2,3).cuda()
    grid = torch.nn.functional.affine_grid(theta, size=torch.Size([N, C, OH, OW]))
    return torch.nn.functional.grid_sample(im, grid)

def heatmap_overlay(heatmap, img, heat_map_ratio=0.4, dim=1):
    # convert heatmap to a real image img (numpy array)
    heatmap = heatmap - np.min(heatmap)
    if np.max(heatmap) != 0: # homogeneous case
        heatmap_img = heatmap / np.max(heatmap)        
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    if np.max(heatmap) != 0: # homogeneous case
        heatmap = heatmap / heatmap.max()
    result = ((heatmap/heatmap.max()) * heat_map_ratio + img * (1-heat_map_ratio)) * dim
    return result
