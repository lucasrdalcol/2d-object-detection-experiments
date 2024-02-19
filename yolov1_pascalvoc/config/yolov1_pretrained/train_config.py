import torch
import os
import torchvision.transforms as transforms

import sys
sys.path.append(os.getenv("TWODOBJECTDETECTION_ROOT"))
from yolov1_pascalvoc.utils.common import *

# Hyperparameters

LEARNING_RATE = 1e-3
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
WEIGHT_DECAY = 0.0005
EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = True
SPLIT_SIZE = 7
NUM_BOXES = 2
NUM_CLASSES = 20
PRE_TRAINED_CNN = True
SHOW_BATCH_IMAGES = False
INPUT_SIZE = (224, 224)
TRANSFORM = Compose([transforms.Resize(INPUT_SIZE), transforms.ToTensor()])
PRINT_NN_SUMMARY = False
LOAD_MODEL_CHECKPOINT = False
LOAD_MODEL_CHECKPOINT_FILENAME = "test.pth.tar"
SAVE_MODEL = True
SAVE_MODEL_FILENAME = "test1_pascal_debug.pth.tar"
PROJECT_NAME = "2d-object-detection-experiments"
PROJECT_DIR = os.path.join(os.getenv("PHD_REPOSITORIES"), PROJECT_NAME)
DATASET_DIR = os.path.join(os.getenv("PHD_DATASETS"), f"{PROJECT_NAME}/PascalVOC_YOLO")
MODELS_DIR = os.path.join(os.getenv("PHD_MODELS"), PROJECT_NAME)
RESULTS_DIR = os.path.join(os.getenv("PHD_RESULTS"), PROJECT_NAME)
SEED = 123
VISUALIZE_RESULTS = False
SAVE_RESULTS = False
SAVE_RESULTS_FOLDER = "test1_pascal_debug"
OVERWRITE_RESULTS_FOLDER = True
DECIMATION_FACTOR = None  # Set to None to use all data
