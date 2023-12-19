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
LOAD_MODEL = False
LOAL_MODEL_FILE = "overfit.pth.tar"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
IMG_DIR = os.path.join(PROJECT_DIR, "data/PascalVOC_YOLO/images")
LABEL_DIR = os.path.join(PROJECT_DIR, "data/PascalVOC_YOLO/labels")
SEED = 123