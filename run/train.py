#!/usr/bin/env python3

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
import time

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.yolo_v1 import *
from data_processing.pascalvoc_yolo import *
from utils.metrics import *
from utils.visualization import *
from loss.yolo_v1_loss import *
from utils.common import *
from utils.training_utils import *
import config.yolov1_config as cfg

# Seed for reproducibility
seed_everything(cfg.SEED)

# transforms for the training data
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


def main():
    # Load model, optimizer and loss function
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20).to(cfg.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    loss_fn = YOLOv1Loss()

    # Load model checkpoint if available
    if cfg.LOAD_MODEL:
        load_checkpoint(torch.load(cfg.LOAD_MODEL_FILENAME), model, optimizer)

    train_dataset = PascalVOCDatasetYOLO(
        os.path.join(cfg.PROJECT_DIR, "data/PascalVOC_YOLO/8examples.csv"),
        img_dir=cfg.IMG_DIR,
        label_dir=cfg.LABEL_DIR,
        transform=transform,
    )

    test_dataset = PascalVOCDatasetYOLO(
        os.path.join(cfg.PROJECT_DIR, "data/PascalVOC_YOLO/test.csv"),
        img_dir=cfg.IMG_DIR,
        label_dir=cfg.LABEL_DIR,
        transform=transform,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    # Train the model
    since = time.time()
    for epoch in range(cfg.EPOCHS):
        print("=" * 50)

        # Each epoch has a training and validation phase
        for phase in ["train"]:
            print("-" * 20)
            print(f"{phase} phase - Epoch {epoch + 1}/{cfg.EPOCHS}")
            print("-" * 20)

            train_epoch(train_dataloader, model, optimizer, loss_fn)

            # Get the predictions and targets bboxes to compute mAP for the training dataset
            train_pred_boxes, train_target_boxes = get_bboxes(
                train_dataloader,
                model,
                iou_threshold=0.5,
                threshold=0.4,
            )

            # Compute mAP for the training data
            train_mean_avg_prec = mean_average_precision(
                train_pred_boxes,
                train_target_boxes,
                iou_threshold=0.5,
                box_format="midpoint",
            )

            print(f"Train mAP: {train_mean_avg_prec}")
        print("=" * 50)

    # Print the time it took to train the model
    time_elapsed = time.time() - since
    print("Training finished!")
    print(
        "Training completed in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    # Save best trained model
    if cfg.SAVE_MODEL:
        # Save best model
        print(f"Saving best model: {cfg.SAVE_MODEL_FILENAME}...")
        torch.save(
            model.state_dict(),
            os.path.join(cfg.PROJECT_DIR, f"trained_models/{cfg.SAVE_MODEL_FILENAME}"),
        )

    # Visualize results
    if cfg.VISUALIZE_RESULTS:
        print(f"Visualizing results...")
        for inputs_x, labels_y in train_dataloader:
            inputs_x = inputs_x.to(cfg.DEVICE)
            for idx in range(8):
                bboxes = cellboxes_to_boxes(model(inputs_x))
                bboxes = non_max_suppression(
                    bboxes[idx],
                    iou_threshold=0.5,
                    prob_threshold=0.4,
                    box_format="midpoint",
                )
                plot_image(inputs_x[idx].permute(1, 2, 0).to("cpu"), bboxes)

            break


if __name__ == "__main__":
    main()
