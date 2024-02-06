#!/usr/bin/env python3

import shutil
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
import time

import sys
import os

sys.path.append(os.getenv("TWODOBJECTDETECTION_ROOT"))
from models.yolo_v1 import *
from data_processing.pascalvoc_yolo import *
from utils.metrics import *
from utils.visualization import *
from loss.yolo_v1_loss import *
from utils.common import *
from utils.training_utils import *
import config.yolov1_infer_config as cfg

# Seed for reproducibility
seed_everything(cfg.SEED)

# transforms for the training data
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


def main():
    # Load model, optimizer and loss function
    model = YOLOv1(split_size=cfg.SPLIT_SIZE, num_boxes=cfg.NUM_BOXES, num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)
    model.load_state_dict(torch.load(os.path.join(cfg.MODELS_DIR, cfg.LOAD_MODEL_FILENAME)))

    # Load the training and validation datasets
    test_dataset = PascalVOCDatasetYOLO(
        os.path.join(cfg.DATASET_DIR, "test.csv"),
        img_dir=os.path.join(cfg.DATASET_DIR, "images"),
        label_dir=os.path.join(cfg.DATASET_DIR, "labels"),
        transform=transform,
        decimation_factor=cfg.DECIMATION_FACTOR
    )

    # Create training and validation dataloaders
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    # Infer the model
    since = time.time()
    print("-" * 20)
    print(f"Inference")
    print("-" * 20)
    test_pred_boxes, test_target_boxes = get_bboxes(
        test_dataloader,
        model,
        iou_threshold=0.5,
        threshold=0.4,
        device=cfg.DEVICE,
        progress_bar=True
    )  # Get the predictions and targets bboxes to compute mAP for the test dataset
    test_mean_avg_prec = mean_average_precision(
        test_pred_boxes,
        test_target_boxes,
        iou_threshold=0.5,
        box_format="midpoint"
    )  # Compute mAP for the test data for inference
    print(f"Inference mAP: {test_mean_avg_prec}")

    # Print the time it took to train the model
    time_elapsed = time.time() - since
    print("Inference finished!")
    print(
        "Inference completed in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    # Visualize and/or save comparison results
    if cfg.VISUALIZE_RESULTS or cfg.SAVE_RESULTS:
        if cfg.SAVE_RESULTS:
            print(f"Saving inference results...")
            results_folder_path = os.path.join(
                cfg.RESULTS_DIR, cfg.SAVE_RESULTS_FOLDER
            )
            if not os.path.exists(results_folder_path):
                os.makedirs(
                    results_folder_path
                )
            else:
                shutil.rmtree(
                    results_folder_path
                )  # Removes all the subdirectories!
                os.makedirs(results_folder_path)
        if cfg.VISUALIZE_RESULTS:
            print(f"Visualizing inference results...")

        for inputs_x, labels_y, filenames in test_dataloader:
            inputs_x = inputs_x.to(cfg.DEVICE)
            labels_y = labels_y.to(cfg.DEVICE)
            for idx in range(inputs_x.shape[0]):
                pred_bboxes = cellboxes_to_boxes(model(inputs_x))
                true_bboxes = cellboxes_to_boxes(labels_y)
                pred_bboxes_idx = non_max_suppression(
                    pred_bboxes[idx],
                    iou_threshold=0.5,
                    prob_threshold=0.4,
                    box_format="midpoint",
                )
                true_bboxes_idx = true_bboxes[idx]
                filename_idx = filenames[idx]
                plt = plot_comparison_image(
                    inputs_x[idx].permute(1, 2, 0).to("cpu"),
                    filename_idx,
                    pred_bboxes_idx,
                    true_bboxes_idx,
                )
                if cfg.SAVE_RESULTS:
                    plt.savefig(
                        os.path.join(results_folder_path,
                            f"{filename_idx.split('.')[0]}.png",
                        )
                    )
                if cfg.VISUALIZE_RESULTS:
                    plt.show()


if __name__ == "__main__":
    main()
