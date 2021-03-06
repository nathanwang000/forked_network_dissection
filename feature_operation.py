import os, cv2, math
from torch.autograd import Variable as V
from scipy.misc import imresize
import numpy as np
import torch
import settings
import time
import util.upsample as upsample
import util.vecquantile as vecquantile
from util.utility import slide2d, slide2d_tensor, convert_image_np, convert_np2var, bilinear_resize
import multiprocessing.pool as pool
from loader.data_loader import load_csv
from loader.data_loader import SegmentationData, SegmentationPrefetcher
from loader.model_loader import getModule

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def hook_inputconv_feature(module, input, output):
    # record last output
    module.output = output.data.cpu().numpy()

def fcn_forward(model, inp_var, size=5, receptive_field=None): 
    '''
       # use for fully connected layers, assumes 4d image
       inp_var: image in pytorch format of shape (bs, c, w, h)
       return: a list of heatmaps in the same order as feature_names
    '''

    heatmap_list = []
    feature_names = settings.FEATURE_NAMES

    # determine receptive_field
    if receptive_field is None:
        receptive_field = settings.RF
    
    # prepare slide2d
    bs, c, w, h = inp_var.shape
    # todo: inefficient one pass is enough for all layers
    for name in feature_names:

        def CAM(inp):
            # change receptive_field to size (w,h) of CAM to kernel size,
            # inp is a pytorch variable
            # out is bs x n_concepts np array
            bs, c, w_in, h_in  = inp.shape
            # use pytorch bilinear interpolation, also only resize if needed
            if w_in != w or h_in != h:
                inp = bilinear_resize(inp, h, w)
            
            _ = model(inp) # doesn't need out
            m = getModule(model, name)
            phix = m.output
            if len(phix.shape) > 1: # assumes have batch dimension
                phix = phix.reshape(phix.shape[0], -1)

            # note: this is slow            
            if settings.CONCEPT_PROJECT: # projection
                return phix.dot(m.P)
            else:
                return m.Pinv.dot(phix.T).T
        
        heatmap = slide2d_tensor(inp_var, (receptive_field, receptive_field), 
                                 pad=math.ceil((receptive_field-1)/2), 
                                 stride=math.floor((w-1)/size), 
                                 transform=CAM)

        heatmap_list.append(heatmap)
    
    return heatmap_list

class FeatureOperator:

    def __init__(self):
        if not os.path.exists(settings.OUTPUT_FOLDER):
            os.makedirs(os.path.join(settings.OUTPUT_FOLDER, 'image'))
        self.data = SegmentationData(settings.DATA_DIRECTORY, categories=settings.CATAGORIES)
        self.loader = SegmentationPrefetcher(self.data,categories=['image'],once=True,batch_size=settings.BATCH_SIZE)
        self.mean = [109.5388,118.6897,124.6901]

    def init_concept_matrix(self, model, mode):
        '''how to initialize concept, see settings.CONCEPT_INIT'''
        print('initializing concepts with mode %s' % mode)
        # make a fake image to gauge how many concepts to generate
        s = settings.IMG_SIZE
        input_var = V(torch.zeros([1, 3, s, s]))
        if settings.GPU:
            input_var = input_var.cuda()
        _  = model(input_var)
        
        for name in settings.FEATURE_NAMES:
            m = getModule(model, name)
            out = m.output # (bs x c x h x w)
            n_concepts = out.shape[1]
            
            if mode is 'identity':
                m.P = np.identity(n_concepts)
                m.Pinv = m.P
            else: # try to load the file, assume npy file
                print('custom class %s' %  mode)
                print('loading basis path from %s' % settings.CONCEPT_PATH)
                m.P = np.load(settings.CONCEPT_PATH)
                m.Pinv = np.linalg.pinv(m.P)
        
    def feature_extraction(self, model=None, memmap=True):
        loader = self.loader
        # extract the max value activaiton for each image
        maxfeatures = [None] * len(settings.FEATURE_NAMES)
        wholefeatures = [None] * len(settings.FEATURE_NAMES) 
        features_size = [None] * len(settings.FEATURE_NAMES) 
        features_size_file = os.path.join(settings.OUTPUT_FOLDER, "feature_size.npy")

        if memmap:
            skip = True
            mmap_files =  [os.path.join(settings.OUTPUT_FOLDER, "%s.mmap" % feature_name)  for feature_name in  settings.FEATURE_NAMES]
            mmap_max_files = [os.path.join(settings.OUTPUT_FOLDER, "%s_max.mmap" % feature_name) for feature_name in settings.FEATURE_NAMES]
            if os.path.exists(features_size_file):
                features_size = np.load(features_size_file)
            else:
                skip = False
            for i, (mmap_file, mmap_max_file) in enumerate(zip(mmap_files,mmap_max_files)):
                if os.path.exists(mmap_file) and os.path.exists(mmap_max_file) and features_size[i] is not None:
                    print('loading features %s' % settings.FEATURE_NAMES[i])
                    wholefeatures[i] = np.memmap(mmap_file, dtype=float,mode='r', shape=tuple(features_size[i]))
                    maxfeatures[i] = np.memmap(mmap_max_file, dtype=float, mode='r', shape=tuple(features_size[i][:2]))
                else:
                    print('file missing, loading from scratch')
                    skip = False
            if skip:
                return wholefeatures, maxfeatures

        # init concept matrix
        self.init_concept_matrix(model, settings.CONCEPT_INIT)
        
        num_batches = (len(loader.indexes) + loader.batch_size - 1) / loader.batch_size
        for batch_idx,batch in enumerate(loader.tensor_batches(bgr_mean=self.mean)):
            del features_blobs[:] # clear all content of feature blob
            input = batch[0]
            batch_size = len(input)
            print('extracting feature from batch %d / %d' % (batch_idx+1, num_batches))
            input = torch.from_numpy(input[:, ::-1, :, :].copy()) # authors: todo preprocess input
            input.div_(255.0 * 0.224)
            if settings.GPU:
                input = input.cuda()
            input_var = V(input,volatile=True)

            logit = model.forward(input_var)
            # create 13x13 forward operations
            if settings.INPUT_CONV:
                # todo: check if need deep copy
                features_blobs.extend(fcn_forward(model, input_var))                
                # im_np = convert_image_np(input_var.data)
                # features_blobs.extend(forward_slide2d(model, im_np))
            
            while np.isnan(logit.data.max()):
                print("nan") #which I have no idea why it will happen (this is author's note)
                del features_blobs[:]
                logit = model.forward(input_var)
                # create 13x13 forward operations
                if settings.INPUT_CONV:
                    # todo: check if need deep copy
                    features_blobs.extend(fcn_forward(model, input_var))   
                    # im_np = convert_image_np(input_var.data)
                    # features_blobs.extend(forward_slide2d(model, im_np))
                
            if maxfeatures[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(features_blobs):
                    size_features = (len(loader.indexes), feat_batch.shape[1])
                    #print(size_features, feat_batch.shape) # 128, 256, 6, 6
                    if memmap:
                        maxfeatures[i] = np.memmap(mmap_max_files[i],dtype=float,mode='w+',shape=size_features)
                    else:
                        maxfeatures[i] = np.zeros(size_features)
                        
            if len(feat_batch.shape) == 4 and wholefeatures[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(features_blobs):
                    size_features = (
                    len(loader.indexes), feat_batch.shape[1], feat_batch.shape[2], feat_batch.shape[3])
                    features_size[i] = size_features
                    if memmap:
                        wholefeatures[i] = np.memmap(mmap_files[i], dtype=float, mode='w+', shape=size_features)
                    else:
                        wholefeatures[i] = np.zeros(size_features)
            np.save(features_size_file, features_size)
            start_idx = batch_idx*settings.BATCH_SIZE
            end_idx = min((batch_idx+1)*settings.BATCH_SIZE, len(loader.indexes))
            for i, feat_batch in enumerate(features_blobs):
                if len(feat_batch.shape) == 4:
                    wholefeatures[i][start_idx:end_idx] = feat_batch
                    maxfeatures[i][start_idx:end_idx] = np.max(np.max(feat_batch,3),2)
                elif len(feat_batch.shape) == 3:
                    maxfeatures[i][start_idx:end_idx] = np.max(feat_batch, 2)
                elif len(feat_batch.shape) == 2:
                    maxfeatures[i][start_idx:end_idx] = feat_batch
        if len(feat_batch.shape) == 2:
            wholefeatures = maxfeatures
        return wholefeatures,maxfeatures

    def quantile_threshold(self, features, savepath=''):
        qtpath = os.path.join(settings.OUTPUT_FOLDER, savepath)
        if savepath and os.path.exists(qtpath):
            return np.load(qtpath)
        print("calculating quantile threshold")
        quant = vecquantile.QuantileVector(depth=features.shape[1], seed=1)
        start_time = time.time()
        last_batch_time = start_time
        batch_size = 64
        for i in range(0, features.shape[0], batch_size):
            batch_time = time.time()
            rate = i / (batch_time - start_time + 1e-15)
            batch_rate = batch_size / (batch_time - last_batch_time + 1e-15)
            last_batch_time = batch_time
            print('Processing quantile index %d: %f %f' % (i, rate, batch_rate))
            batch = features[i:i + batch_size]
            batch = np.transpose(batch, axes=(0, 2, 3, 1)).reshape(-1, features.shape[1])
            quant.add(batch)
        ret = quant.readout(1000)[:, int(1000 * (1-settings.QUANTILE)-1)]
        if savepath:
            np.save(qtpath, ret)
        return ret
        # return np.percentile(features,100*(1 - settings.QUANTILE),axis=axis)

    @staticmethod
    def tally_job(args):
        features, data, threshold, tally_labels, tally_units, tally_units_cat, tally_both, start, end = args
        units = features.shape[1]
        size_RF = (settings.IMG_SIZE / features.shape[2], settings.IMG_SIZE / features.shape[3])
        fieldmap = ((0, 0), size_RF, size_RF)
        pd = SegmentationPrefetcher(data, categories=data.category_names(),
                                    once=True, batch_size=settings.TALLY_BATCH_SIZE,
                                    ahead=settings.TALLY_AHEAD, start=start, end=end)
        count = start
        start_time = time.time()
        last_batch_time = start_time
        for batch in pd.batches():
            batch_time = time.time()
            rate = (count - start) / (batch_time - start_time + 1e-15)
            batch_rate = len(batch) / (batch_time - last_batch_time + 1e-15)
            last_batch_time = batch_time

            print('labelprobe image index %d, items per sec %.4f, %.4f' % (count, rate, batch_rate))

            for concept_map in batch:
                count += 1
                img_index = concept_map['i']
                scalars, pixels = [], []
                for cat in data.category_names():
                    label_group = concept_map[cat]
                    shape = np.shape(label_group)
                    if len(shape) % 2 == 0:
                        label_group = [label_group]
                    if len(shape) < 2:
                        scalars += label_group
                    else:
                        pixels.append(label_group)
                for scalar in scalars:
                    tally_labels[scalar] += concept_map['sh'] * concept_map['sw']
                if pixels:
                    pixels = np.concatenate(pixels)
                    tally_label = np.bincount(pixels.ravel())
                    if len(tally_label) > 0:
                        tally_label[0] = 0
                    tally_labels[:len(tally_label)] += tally_label

                for unit_id in range(units):
                    feature_map = features[img_index][unit_id]
                    if feature_map.max() > threshold[unit_id]:
                        mask = imresize(feature_map, (concept_map['sh'], concept_map['sw']), mode='F') # 32 bit floating point pixel
                        #reduction = int(round(settings.IMG_SIZE / float(concept_map['sh'])))
                        #mask = upsample.upsampleL(fieldmap, feature_map, shape=(concept_map['sh'], concept_map['sw']), reduction=reduction)
                        indexes = np.argwhere(mask > threshold[unit_id])

                        tally_units[unit_id] += len(indexes)
                        if len(pixels) > 0:
                            tally_bt = np.bincount(pixels[:, indexes[:, 0], indexes[:, 1]].ravel())
                            if len(tally_bt) > 0:
                                tally_bt[0] = 0
                            tally_cat = np.dot(tally_bt[None,:], data.labelcat[:len(tally_bt), :])[0]
                            tally_both[unit_id,:len(tally_bt)] += tally_bt
                        for scalar in scalars:
                            tally_cat += data.labelcat[scalar]
                            tally_both[unit_id, scalar] += len(indexes)
                        tally_units_cat[unit_id] += len(indexes) * (tally_cat > 0)


    def tally(self, features, threshold, savepath=''):
        csvpath = os.path.join(settings.OUTPUT_FOLDER, savepath)
        if savepath and os.path.exists(csvpath):
            return load_csv(csvpath)

        units = features.shape[1]
        labels = len(self.data.label)
        categories = self.data.category_names()
        tally_both = np.zeros((units,labels),dtype=np.float64)
        tally_units = np.zeros(units,dtype=np.float64)
        tally_units_cat = np.zeros((units,len(categories)), dtype=np.float64)
        tally_labels = np.zeros(labels,dtype=np.float64)

        if settings.PARALLEL > 1:
            psize = int(np.ceil(float(self.data.size()) / settings.PARALLEL))
            ranges = [(s, min(self.data.size(), s + psize)) for s in range(0, self.data.size(), psize) if
                      s < self.data.size()]
            params = [(features, self.data, threshold, tally_labels, tally_units, tally_units_cat, tally_both) + r for r in ranges]
            threadpool = pool.ThreadPool(processes=settings.PARALLEL)
            threadpool.map(FeatureOperator.tally_job, params)
        else:
            FeatureOperator.tally_job((features, self.data, threshold, tally_labels, tally_units, tally_units_cat, tally_both, 0, self.data.size()))

        primary_categories = self.data.primary_categories_per_index()
        tally_units_cat = np.dot(tally_units_cat, self.data.labelcat.T)
        iou = tally_both / (tally_units_cat + tally_labels[np.newaxis,:] - tally_both + 1e-10)
        pciou = np.array([iou * (primary_categories[np.arange(iou.shape[1])] == ci)[np.newaxis, :] for ci in range(len(self.data.category_names()))])
        label_pciou = pciou.argmax(axis=2)
        name_pciou = [
            [self.data.name(None, j) for j in label_pciou[ci]]
            for ci in range(len(label_pciou))]
        score_pciou = pciou[
            np.arange(pciou.shape[0])[:, np.newaxis],
            np.arange(pciou.shape[1])[np.newaxis, :],
            label_pciou]
        bestcat_pciou = score_pciou.argsort(axis=0)[::-1]
        ordering = score_pciou.max(axis=0).argsort()[::-1]
        rets = [None] * len(ordering)

        for i,unit in enumerate(ordering):
            # Top images are top[unit]
            bestcat = bestcat_pciou[0, unit]
            data = {
                'unit': (unit + 1),
                'category': categories[bestcat],
                'label': name_pciou[bestcat][unit],
                'score': score_pciou[bestcat][unit]
            }
            for ci, cat in enumerate(categories):
                label = label_pciou[ci][unit]
                data.update({
                    '%s-label' % cat: name_pciou[ci][unit],
                    '%s-truth' % cat: tally_labels[label],
                    '%s-activation' % cat: tally_units_cat[unit, label],
                    '%s-intersect' % cat: tally_both[unit, label],
                    '%s-iou' % cat: score_pciou[ci][unit]
                })
            rets[i] = data

        if savepath:
            import csv
            csv_fields = sum([[
                '%s-label' % cat,
                '%s-truth' % cat,
                '%s-activation' % cat,
                '%s-intersect' % cat,
                '%s-iou' % cat] for cat in categories],
                ['unit', 'category', 'label', 'score'])
            with open(csvpath, 'w') as f:
                writer = csv.DictWriter(f, csv_fields)
                writer.writeheader()
                for i in range(len(ordering)):
                    writer.writerow(rets[i])
        return rets

