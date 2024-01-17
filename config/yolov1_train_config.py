import torch
import os

# Hyperparameters

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
SPLIT_SIZE = 7
NUM_BOXES = 2
NUM_CLASSES = 20
LOAD_MODEL_CHECKPOINT = False
LOAD_MODEL_CHECKPOINT_FILENAME = "test.pth.tar"
SAVE_MODEL = True
SAVE_MODEL_FILENAME = "overfit_wandb.pth.tar"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
IMG_DIR = os.path.join(PROJECT_DIR, "data/PascalVOC_YOLO/images")
LABEL_DIR = os.path.join(PROJECT_DIR, "data/PascalVOC_YOLO/labels")
SEED = 123
VISUALIZE_RESULTS = False
SAVE_RESULTS = False
SAVE_RESULTS_FOLDER = "test"
OVERWRITE_RESULTS_FOLDER = True
DECIMATION_FACTOR = None # Set to None to use all data