import settings
import torch
import torchvision

def getModule(model, name):
    if len(name.split('.')) == 1:
        m = model._modules.get(name)
    else:
        name, id = name.split('.')
        m = model._modules.get(name)[int(id)]
    return m
    
def loadmodel(hook_fn):
    if settings.MODEL_FILE is None:
        model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
    else:
        checkpoint = torch.load(settings.MODEL_FILE)
        if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
            model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
            if settings.MODEL_PARALLEL:
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                    'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            model = checkpoint
    for name in settings.FEATURE_NAMES:
        m = getModule(model, name)
        m.register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model
