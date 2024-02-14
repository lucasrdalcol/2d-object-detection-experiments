import torch
import os
import torchvision.transforms as transforms

import sys
sys.path.append(os.getenv("TWODOBJECTDETECTION_ROOT"))
from yolov1_nuimages.utils.common import *

# Hyperparameters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
PIN_MEMORY = True
BATCH_SIZE = 16
SPLIT_SIZE = 7
NUM_BOXES = 2
NUM_CLASSES = 25
PRE_TRAINED_CNN = False
INPUT_SIZE = (448, 448)
TRANSFORM = Compose([transforms.Resize(INPUT_SIZE), transforms.ToTensor()])
PRINT_NN_SUMMARY = False
LOAD_MODEL = False
LOAD_MODEL_FILENAME = "overfit.pth.tar"
PROJECT_NAME = "2d-object-detection-experiments"
PROJECT_DIR = os.path.join(os.getenv("PHD_REPOSITORIES"), PROJECT_NAME)
DATASET_DIR = os.path.join(os.getenv("PHD_DATASETS"), f"{PROJECT_NAME}/PascalVOC_YOLO")
MODELS_DIR = os.path.join(os.getenv("PHD_MODELS"), PROJECT_NAME)
RESULTS_DIR = os.path.join(os.getenv("PHD_RESULTS"), PROJECT_NAME)
SEED = 123
VISUALIZE_RESULTS = False
SAVE_RESULTS = True
SAVE_RESULTS_FOLDER = "infer_overfit_experiment"
OVERWRITE_RESULTS_FOLDER = True
DECIMATION_FACTOR = None  # Set to None to use all data
