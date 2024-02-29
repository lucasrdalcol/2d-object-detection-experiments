import torch
import os
import torchvision.transforms as transforms

import sys

sys.path.append(os.getenv("TWODOBJECTDETECTION_ROOT"))
from yolov1_nuimages.utils.common import *

# Hyperparameters

LEARNING_RATE = 2e-5
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
WEIGHT_DECAY = 0.001
EPOCHS = 135
NUM_WORKERS = 14
PIN_MEMORY = True if "cuda" in DEVICE else False
SPLIT_SIZE = 7
NUM_BOXES = 2
NUM_CLASSES = 25
PRE_TRAINED_CNN = True
INPUT_SIZE = (224, 224)
FCL_SIZE = 512
DROPOUT = 0.5
SHOW_BATCH_IMAGES = False
PRINT_NN_SUMMARY = False
LOAD_MODEL_CHECKPOINT = False
LOAD_MODEL_CHECKPOINT_FILENAME = "test.pth.tar"
SAVE_MODEL = True
SAVE_MODEL_FILENAME = "test_decimation.pth"
PROJECT_NAME = "2d-object-detection-experiments"
PROJECT_DIR = os.path.join(os.getenv("PHD_REPOSITORIES"), PROJECT_NAME)
DATASET_DIR = os.path.join(
    os.getenv("PHD_DATASETS"), f"{PROJECT_NAME}/nuImages/nuimages-v1.0"
)
TRAIN_DATASET_VERSION = "v1.0-train"
VAL_DATASET_VERSION = "v1.0-val"
MODELS_DIR = os.path.join(os.getenv("PHD_MODELS"), PROJECT_NAME)
RESULTS_DIR = os.path.join(os.getenv("PHD_RESULTS"), PROJECT_NAME)
SEED = 123
VISUALIZE_RESULTS = False
SAVE_RESULTS = False
SAVE_RESULTS_FOLDER = "test5"
OVERWRITE_RESULTS_FOLDER = True
TRAIN_DECIMATION_FACTOR = 0.5  # Set to None to use all data
VAL_DECIMATION_FACTOR = 0.95  # Set to None to use all data
