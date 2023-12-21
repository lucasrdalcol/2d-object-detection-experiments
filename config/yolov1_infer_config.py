import torch
import os

# Hyperparameters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
PIN_MEMORY = True
BATCH_SIZE = 16
SPLIT_SIZE = 7
NUM_BOXES = 2
NUM_CLASSES = 20
LOAD_MODEL = False
LOAD_MODEL_FILENAME = "overfit.pth.tar"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
IMG_DIR = os.path.join(PROJECT_DIR, "data/PascalVOC_YOLO/images")
LABEL_DIR = os.path.join(PROJECT_DIR, "data/PascalVOC_YOLO/labels")
SEED = 123
VISUALIZE_RESULTS = False
SAVE_RESULTS = True
SAVE_RESULTS_FOLDER = "infer_overfit_experiment"
OVERWRITE_RESULTS_FOLDER = True
DECIMATION_FACTOR = None # Set to None to use all data
