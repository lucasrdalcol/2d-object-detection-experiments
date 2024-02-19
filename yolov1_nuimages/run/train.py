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
import torchinfo
import matplotlib.pyplot as plt

import sys
import os
import importlib
import importlib.util

sys.path.append(os.getenv("TWODOBJECTDETECTION_ROOT"))
from yolov1_nuimages.models.yolo_v1 import *
from yolov1_nuimages.models.yolo_v1_pre_trained import *
from yolov1_nuimages.data_processing.nuimages_yolo import *
from yolov1_nuimages.utils.metrics import *
from yolov1_nuimages.utils.visualization import *
from yolov1_nuimages.loss.yolo_v1_loss import *
from yolov1_nuimages.utils.common import *
from yolov1_nuimages.utils.training_utils import *

import yolov1_nuimages.config.train_config_master as train_config_master

cfg = importlib.import_module(train_config_master.CONFIG_FILE)

# Seed for reproducibility
seed_everything(cfg.SEED)


def main():
    # Config dict creation
    spec = importlib.util.spec_from_file_location(
        "config", os.path.abspath(cfg.__file__)
    )
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config_dict = {
        name: getattr(config_module, name)
        for name in dir(config_module)
        if not name.startswith("__") and name.isupper()
    }

    # Load model, optimizer and loss function
    if not cfg.PRE_TRAINED_CNN:
        model = YOLOv1(
            split_size=cfg.SPLIT_SIZE,
            num_boxes=cfg.NUM_BOXES,
            num_classes=cfg.NUM_CLASSES,
        ).to(cfg.DEVICE)
    else:
        model = YOLOv1PreTrained(
            split_size=cfg.SPLIT_SIZE,
            num_boxes=cfg.NUM_BOXES,
            num_classes=cfg.NUM_CLASSES,
        ).to(cfg.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    loss_fn = YOLOv1Loss()

    print(
        f"The model is in device: {next(model.parameters()).device}"
    )  # Check if the model is in the GPU

    # Print summary of the model
    if cfg.PRINT_NN_SUMMARY:
        print(model)
        torchinfo.summary(
            model,
            input_size=(cfg.BATCH_SIZE, 3, cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1]),
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
                "mult_adds",
            ),
            verbose=1,
        )

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="YOLOv1_NuImages",
        name="pretrained_test3_resnet18",
        # track hyperparameters and run metadata
        config=config_dict,
        # mode="disabled",
    )

    wandb.watch(model)

    # Load model checkpoint if available
    if cfg.LOAD_MODEL_CHECKPOINT:
        load_checkpoint(
            torch.load(cfg.LOAD_MODEL_CHECKPOINT_FILENAME), model, optimizer
        )

    # Load the training and validation datasets
    nuimages_train = NuImages(
        dataroot=cfg.DATASET_DIR, version="v1.0-train", verbose=True, lazy=False
    )
    nuimages_val = NuImages(
        dataroot=cfg.DATASET_DIR, version="v1.0-val", verbose=True, lazy=False
    )

    train_dataset = NuImagesDatasetYOLO(
        nuimages_train,
        transform=cfg.TRANSFORM,
        remove_empty=True,
        split_size=cfg.SPLIT_SIZE,
        num_boxes=cfg.NUM_BOXES,
        num_classes=cfg.NUM_CLASSES,
    )

    val_dataset = NuImagesDatasetYOLO(
        nuimages_val,
        transform=cfg.TRANSFORM,
        remove_empty=True,
        split_size=cfg.SPLIT_SIZE,
        num_boxes=cfg.NUM_BOXES,
        num_classes=cfg.NUM_CLASSES,
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

    # Visualize the transformed images for a batch
    if cfg.SHOW_BATCH_IMAGES:
        input_images, label_matrices, filenames = next(iter(val_dataloader))
        for i in range(len(input_images)):
            # Get the i-th image
            image = (
                input_images[i].cpu().numpy().transpose(1, 2, 0)
            )  # Change from (channels, height, width) to (height, width, channels)
            print(image)
            # Create a new figure
            plt.figure()
            # Plot the image
            plt.imshow(image)
            plt.axis("on")  # Turn on the axes
            # Add a title
            plt.title(filenames[i])
            # Show the plot
            plt.show()

    # Train the model
    _ = next(
        iter(train_dataloader)
    )  # Load the first batch to check if everything is working
    _ = next(iter(val_dataloader))
    since = time.time()
    for epoch in range(cfg.EPOCHS):
        print("=" * 50)

        # Each epoch has a training and validation phase
        # Training phase
        print("-" * 20)
        print(f"Training phase - Epoch {epoch + 1}/{cfg.EPOCHS}")
        print("-" * 20)
        train_loss = train_epoch(
            train_dataloader, model, optimizer, loss_fn, device=cfg.DEVICE
        )

        # Validation phase
        print("-" * 20)
        print(f"Validation phase - Epoch {epoch + 1}/{cfg.EPOCHS}")
        print("-" * 20)
        val_loss, val_mean_avg_prec = validate_epoch(
            val_dataloader,
            model,
            loss_fn,
            iou_threshold=0.5,
            prob_threshold=0.4,
            device=cfg.DEVICE,
        )
        print(f"Val mAP: {val_mean_avg_prec}")

        # Log metrics to wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "train": {"loss": train_loss},
                "val": {"mAP": val_mean_avg_prec, "loss": val_loss},
            }
        )

        print("=" * 50)

    # Print the time it took to train the model
    time_elapsed = time.time() - since
    hours = int(time_elapsed // 3600)
    minutes = int((time_elapsed % 3600) // 60)
    seconds = int(time_elapsed % 60)
    print("Training finished!")
    print("Training completed in {}h {}m {}s".format(hours, minutes, seconds))

    # Save best trained model
    if cfg.SAVE_MODEL:
        # Save best model
        print(f"Saving best model: {cfg.SAVE_MODEL_FILENAME}...")
        torch.save(
            model.state_dict(),
            os.path.join(cfg.MODELS_DIR, cfg.SAVE_MODEL_FILENAME),
        )

    # Visualize and/or save comparison results locally
    if cfg.VISUALIZE_RESULTS or cfg.SAVE_RESULTS:
        if cfg.SAVE_RESULTS:
            print(f"Saving validation results...")
            results_folder_path = os.path.join(cfg.RESULTS_DIR, cfg.SAVE_RESULTS_FOLDER)
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
                plt_image = plot_comparison_image(
                    inputs_x[idx].permute(1, 2, 0).to("cpu"),
                    filename_idx,
                    pred_bboxes_idx,
                    true_bboxes_idx,
                )
                if cfg.SAVE_RESULTS:
                    plt_image.savefig(
                        os.path.join(
                            results_folder_path,
                            f"{filename_idx.split('.')[0]}.png",
                        )
                    )
                if cfg.VISUALIZE_RESULTS:
                    plt_image.show()

    # Log the images and model predictions to wandb
    # this is the order in which my classes will be displayed
    category_id_to_name = {}
    for idx, category in enumerate(nuimages_train.category):
        category_id_to_name[idx] = category["name"]

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
                category_id_to_name,
            )
            wandb_truebbox_image = wandb_bounding_boxes(
                inputs_x[idx].permute(1, 2, 0).to("cpu").numpy(),
                filename_idx,
                true_bboxes_idx,
                category_id_to_name,
            )

            wandb.log(
                {
                    "predicted_bboxes": wandb_predbbox_image,
                    "true_bboxes": wandb_truebbox_image,
                }
            )


if __name__ == "__main__":
    main()
