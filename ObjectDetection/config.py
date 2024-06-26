from easydict import EasyDict as edict
import os

__C                                        = edict()

# config can be used by: `from config import cfg`
cfg                                        = __C

# configuration for YOLO_v5 model
__C.MODEL                                  = edict()
__C.MODEL.REPO                             = 'https://github.com/ultralytics/yolov5.git'
__C.MODEL.DIR                              = 'yolo_v5'
# URL of pre-trained YOLO_v5 models
__C.MODEL.URL                              = 'https://github.com/ultralytics/yolov5/releases/download/v6.1/'
# configuration for datasets (train, test, validation)
__C.DATASET_DIR                            = 'data'
__C.DATASET                                = 'data/dataset.yaml'
__C.TEST                                   = 'data/test_dataset.yaml'
__C.OID_URL                                = 'https://storage.googleapis.com/openimages/2018_04/'


# configuration for bcolors
__C.BCOLORS                                = edict()
__C.BCOLORS.HEADER                         = '\033[95m'
__C.BCOLORS.INFO                           = '    [INFO] | '
__C.BCOLORS.OKBLUE                         = '\033[94m[DOWNLOAD] | '
__C.BCOLORS.WARNING                        = '\033[93m    [WARN] | '
__C.BCOLORS.FAIL                           = '\033[91m   [ERROR] | '
__C.BCOLORS.OKGREEN                        = '\033[92m'
__C.BCOLORS.ENDC                           = '\033[0m'
