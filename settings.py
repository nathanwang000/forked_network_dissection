######### global settings  #########
INPUT_CONV = True                           # input convolution
GPU = True                                  # running on GPU is highly suggested
TEST_MODE = False                           # turning on the testmode means the code will run on a small dataset.
CLEAN = False                               # set to "True" if you want to clean the temporary large files after generating result
MODEL = 'alexnet'                           # model arch: resnet18, alexnet, resnet50, densenet161
DATASET = 'imagenet' #'places365'           # model trained on: places365 or imagenet
QUANTILE = 0.005                            # the threshold used for activation
SEG_THRESHOLD = 0.04                        # the threshold used for visualization # never used in this codebase
SCORE_THRESHOLD = 0.04                      # the threshold used for IoU score (in HTML file)
TOPN = 10                                   # to show top N image with highest activation for each unit
PARALLEL = 1                                # how many process is used for tallying (Experiments show that 1 is the fastest)
RF = 224                                    # receptive field size
CATAGORIES = ["object", "part","scene","texture","color","material"] # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"

'''how to initialize concept
'identity' for using its original basis * 
'random' for random concept initialization *
'broden' for broden concept initialization
'supervise' for supervised learning and taking W' for concept initialization
'W_trans' for taking last layer W' and complete W' for concept initialization *
'ortho' for orthogonal concept initialization
custom names use preloaded concept class with path specified CONCEPT_PATH *
'''
# this path is only used if concept-init uses custom names
# alexnet init with random tiny imagenet (200 categories): '/data/jiaxuan/faithful_interpretation/webapp/basis/20180601062504.npy'
# alexnet init with random broden dataset: 'basis/alexnet_broden_random_0.npy'
# alexnet init with random broden dataset using dummy cat detector neuron 1145 'basis/dummy_1145.npy'
##############***************** change both when used!!!!!!!!!!!!!!!!!!*****************#############################
CONCEPT_INIT = 'random_broden1'
CONCEPT_PATH = 'basis/alexnet_broden_random_1.npy'
# solving for inverse or projection
CONCEPT_PROJECT = True

# result will be stored in this folder
OUTPUT_FOLDER = "result/pytorch_" + MODEL + "_"+DATASET + "/" + \
                (('T' + str(RF) + CONCEPT_INIT + ('_p' if CONCEPT_PROJECT else ''))
                 if INPUT_CONV else 'F')

########### sub settings ###########
# In most of the case, you don't have to change them.
# DATA_DIRECTORY: where broaden dataset locates
# IMG_SIZE: image size, alexnet use 227x227
# NUM_CLASSES: how many labels in final prediction
# FEATURE_NAMES: the array of layer where features will be extracted
# MODEL_FILE: the model file to be probed, "None" means the pretrained model in torchvision
# MODEL_PARALLEL: some model is trained in multi-GPU, so there is another way to load them.
# WORKERS: how many workers are fetching images
# BATCH_SIZE: batch size used in feature extraction
# TALLY_BATCH_SIZE: batch size used in tallying
# INDEX_FILE: if you turn on the TEST_MODE, actually you should provide this file on your own

if MODEL != 'alexnet':
    DATA_DIRECTORY = 'dataset/broden1_224'
    IMG_SIZE = 224
else:
    DATA_DIRECTORY = 'dataset/broden1_227'
    IMG_SIZE = 227

if DATASET == 'places365':
    NUM_CLASSES = 365
elif DATASET == 'imagenet':
    NUM_CLASSES = 1000

if MODEL == 'alexnet':
    MODEL_FILE = None
    if INPUT_CONV:
        FEATURE_NAMES = ['classifier.5'] # last layer of neurons
    else:
        FEATURE_NAMES = ['features.12'] # conv layer 12
elif MODEL == 'resnet18':
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/resnet18_places365.pth.tar'
        MODEL_PARALLEL = True
    elif DATASET == 'imagenet':
        MODEL_FILE = None
        MODEL_PARALLEL = False
elif MODEL == 'densenet161':
    FEATURE_NAMES = ['features']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/whole_densenet161_places365_python36.pth.tar'
        MODEL_PARALLEL = False
elif MODEL == 'resnet50':
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/whole_resnet50_places365_python36.pth.tar'
        MODEL_PARALLEL = False

if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 4
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
    INDEX_FILE = 'index_sm.csv'
    OUTPUT_FOLDER += "_test"
else:
    WORKERS = 24 #12
    BATCH_SIZE = 128
    TALLY_BATCH_SIZE = 16
    TALLY_AHEAD = 4
    INDEX_FILE = 'index.csv'
