#!/usr/bin/env python3

import shutil
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
import wandb
import random

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
import config.yolov1_train_config as cfg

# Seed for reproducibility
seed_everything(cfg.SEED)

# transforms for the training data
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


def main():
    # Load model, optimizer and loss function
    model = YOLOv1(
        split_size=cfg.SPLIT_SIZE, num_boxes=cfg.NUM_BOXES, num_classes=cfg.NUM_CLASSES
    ).to(cfg.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    loss_fn = YOLOv1Loss()

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="YOLOv1",
        name="overfit_test",
        # track hyperparameters and run metadata
        config={
            "learning_rate": cfg.LEARNING_RATE,
            "device": cfg.DEVICE,
            "batch_size": cfg.BATCH_SIZE,
            "weight_decay": cfg.WEIGHT_DECAY,
            "epochs": cfg.EPOCHS,
            "num_workers": cfg.NUM_WORKERS,
            "pin_memory": cfg.PIN_MEMORY,
            "split_size": cfg.SPLIT_SIZE,
            "num_boxes": cfg.NUM_BOXES,
            "num_classes": cfg.NUM_CLASSES,
            "load_model_checkpoint": cfg.LOAD_MODEL_CHECKPOINT,
            "load_model_checkpoint_filename": cfg.LOAD_MODEL_CHECKPOINT_FILENAME,
            "save_model": cfg.SAVE_MODEL,
            "save_model_filename": cfg.SAVE_MODEL_FILENAME,
            "project_dir": cfg.PROJECT_DIR,
            "img_dir": cfg.IMG_DIR,
            "label_dir": cfg.LABEL_DIR,
            "seed": cfg.SEED,
            "visualize_results": cfg.VISUALIZE_RESULTS,
            "save_results": cfg.SAVE_RESULTS,
            "save_results_folder": cfg.SAVE_RESULTS_FOLDER,
            "overwrite_results_folder": cfg.OVERWRITE_RESULTS_FOLDER,
            "decimation_factor": cfg.DECIMATION_FACTOR,
        },
    )

    wandb.watch(model)

    # Load model checkpoint if available
    if cfg.LOAD_MODEL_CHECKPOINT:
        load_checkpoint(
            torch.load(cfg.LOAD_MODEL_CHECKPOINT_FILENAME), model, optimizer
        )

    # Load the training and validation datasets
    train_dataset = PascalVOCDatasetYOLO(
        os.path.join(cfg.PROJECT_DIR, "data/PascalVOC_YOLO/100examples.csv"),
        img_dir=cfg.IMG_DIR,
        label_dir=cfg.LABEL_DIR,
        transform=transform,
        decimation_factor=cfg.DECIMATION_FACTOR,
    )

    val_dataset = PascalVOCDatasetYOLO(
        os.path.join(cfg.PROJECT_DIR, "data/PascalVOC_YOLO/100examples.csv"),
        img_dir=cfg.IMG_DIR,
        label_dir=cfg.LABEL_DIR,
        transform=transform,
        decimation_factor=cfg.DECIMATION_FACTOR,
    )

    # Create training and validation dataloaders
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
        shuffle=False,
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
        train_loss = train_epoch(train_dataloader, model, optimizer, loss_fn)
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
        val_loss = validate_epoch(val_dataloader, model, loss_fn)
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

        # Log metrics to wandb
        wandb.log(
            {
                "train": {"mAP": train_mean_avg_prec, "loss": train_loss},
                "val": {"mAP": val_mean_avg_prec, "loss": val_loss},
            }
        )

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

    # Visualize and/or save comparison results locally
    if cfg.VISUALIZE_RESULTS or cfg.SAVE_RESULTS:
        if cfg.SAVE_RESULTS:
            print(f"Saving validation results...")
            results_folder_path = os.path.join(
                cfg.PROJECT_DIR, f"results/{cfg.SAVE_RESULTS_FOLDER}"
            )
            if not os.path.exists(results_folder_path):
                os.makedirs(results_folder_path)
            else:
                shutil.rmtree(results_folder_path)  # Removes all the subdirectories!
                os.makedirs(results_folder_path)
        if cfg.VISUALIZE_RESULTS:
            print(f"Visualizing validation results...")

        for inputs_x, labels_y, filenames in val_dataloader:
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
                        os.path.join(
                            results_folder_path,
                            f"{filename_idx.split('.')[0]}.png",
                        )
                    )
                if cfg.VISUALIZE_RESULTS:
                    plt.show()

    # Log the images and model predictions to wandb
    # this is the order in which my classes will be displayed
    class_id_to_label = {
        0: "aeroplane",
        1: "bicycle",
        2: "bird",
        3: "boat",
        4: "bottle",
        5: "bus",
        6: "car",
        7: "cat",
        8: "chair",
        9: "cow",
        10: "diningtable",
        11: "dog",
        12: "horse",
        13: "motorbike",
        14: "person",
        15: "pottedplant",
        16: "sheep",
        17: "sofa",
        18: "train",
        19: "tvmonitor",
    }

    for inputs_x, labels_y, filenames in val_dataloader:
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

            # log to wandb: raw image, predictions, and dictionary of class labels for each class id
            wandb_predbbox_image = wandb_bounding_boxes(
                inputs_x[idx].permute(1, 2, 0).to("cpu").numpy(),
                filename_idx,
                pred_bboxes_idx,
                class_id_to_label,
            )
            wandb_truebbox_image = wandb_bounding_boxes(
                inputs_x[idx].permute(1, 2, 0).to("cpu").numpy(),
                filename_idx,
                true_bboxes_idx,
                class_id_to_label,
            )

            wandb.log(
                {
                    "predicted_bboxes": wandb_predbbox_image,
                    "true_bboxes": wandb_truebbox_image,
                }
            )


if __name__ == "__main__":
    main()
