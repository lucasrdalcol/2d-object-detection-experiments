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
        os.path.join(cfg.PROJECT_DIR, "data/PascalVOC_YOLO/100examples.csv"),
        img_dir=cfg.IMG_DIR,
        label_dir=cfg.LABEL_DIR,
        transform=transform,
    )

    val_dataset = PascalVOCDatasetYOLO(
        os.path.join(cfg.PROJECT_DIR, "data/PascalVOC_YOLO/100examples.csv"),
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

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    # Train the model
    since = time.time()
    for epoch in range(cfg.EPOCHS):
        print("=" * 50)

        # Each epoch has a training and validation phase
        # Training phase
        print("-" * 20)
        print(f"Training phase - Epoch {epoch + 1}/{cfg.EPOCHS}")
        print("-" * 20)
        train_epoch(train_dataloader, model, optimizer, loss_fn)
        train_pred_boxes, train_target_boxes = get_bboxes(
            train_dataloader,
            model,
            iou_threshold=0.5,
            threshold=0.4,
        )  # Get the predictions and targets bboxes to compute mAP for the training dataset
        train_mean_avg_prec = mean_average_precision(
            train_pred_boxes,
            train_target_boxes,
            iou_threshold=0.5,
            box_format="midpoint",
        )  # Compute mAP for the training data
        print(f"Train mAP: {train_mean_avg_prec}")

        # Validation phase
        print("-" * 20)
        print(f"Validation phase - Epoch {epoch + 1}/{cfg.EPOCHS}")
        print("-" * 20)
        validate_epoch(val_dataloader, model, loss_fn)
        val_pred_boxes, val_target_boxes = get_bboxes(
            val_dataloader,
            model,
            iou_threshold=0.5,
            threshold=0.4,
        )
        val_mean_avg_prec = mean_average_precision(
            val_pred_boxes,
            val_target_boxes,
            iou_threshold=0.5,
            box_format="midpoint",
        )
        print(f"Val mAP: {val_mean_avg_prec}")

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

    # Visualize and/or save comparison results
    if cfg.VISUALIZE_RESULTS or cfg.SAVE_RESULTS:
        if cfg.SAVE_RESULTS:
            print(f"Saving validation results...")
        if cfg.VISUALIZE_RESULTS:
            print(f"Visualizing validation results...")

        for inputs_x, labels_y, img_name in val_dataloader:
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
                img_name_idx = img_name[idx]
                plt = plot_comparison_image(
                    inputs_x[idx].permute(1, 2, 0).to("cpu"),
                    img_name_idx,
                    pred_bboxes_idx,
                    true_bboxes_idx,
                )
                if cfg.SAVE_RESULTS:
                    os.makedirs(
                        os.path.join(
                            cfg.PROJECT_DIR, f"results/{cfg.SAVE_RESULTS_FOLDER}"
                        ),
                        exist_ok=cfg.OVERWRITE_RESULTS_FOLDER,
                    )
                    plt.savefig(
                        os.path.join(
                            cfg.PROJECT_DIR,
                            f"results/{cfg.SAVE_RESULTS_FOLDER}/{img_name_idx.split('.')[0]}.png",
                        )
                    )
                if cfg.VISUALIZE_RESULTS:
                    plt.show()


if __name__ == "__main__":
    main()
