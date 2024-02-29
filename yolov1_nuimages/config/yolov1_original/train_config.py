import torch
import os
import torchvision.transforms as transforms

import sys
sys.path.append(os.getenv("TWODOBJECTDETECTION_ROOT"))
from yolov1_nuimages.utils.common import *

# Hyperparameters

LEARNING_RATE = 2e-5
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 12
PIN_MEMORY = True
SPLIT_SIZE = 7
NUM_BOXES = 2
NUM_CLASSES = 25
PRE_TRAINED_CNN = False
INPUT_SIZE = (448, 448)
FCL_SIZE = 512
DROPOUT = 0.0
SHOW_BATCH_IMAGES = False
PRINT_NN_SUMMARY = False
LOAD_MODEL_CHECKPOINT = False
LOAD_MODEL_CHECKPOINT_FILENAME = "test.pth.tar"
SAVE_MODEL = True
SAVE_MODEL_FILENAME = "dataset_test.pth.tar"
PROJECT_NAME = "2d-object-detection-experiments"
PROJECT_DIR = os.path.join(os.getenv("PHD_REPOSITORIES"), PROJECT_NAME)
DATASET_DIR = os.path.join(
    os.getenv("PHD_DATASETS"), f"{PROJECT_NAME}/nuImages/nuimages-v1.0"
)
TRAIN_DATASET_VERSION = "v1.0-mini"
VAL_DATASET_VERSION = "v1.0-mini"
MODELS_DIR = os.path.join(os.getenv("PHD_MODELS"), PROJECT_NAME)
RESULTS_DIR = os.path.join(os.getenv("PHD_RESULTS"), PROJECT_NAME)
# SEED = 99
SEED = 123
VISUALIZE_RESULTS = False
SAVE_RESULTS = True
SAVE_RESULTS_FOLDER = "test3"
OVERWRITE_RESULTS_FOLDER = True
DECIMATION_FACTOR = None  # Set to None to use all data
