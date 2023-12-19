#!/usr/bin/env python3

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from models.yolo_v1 import YOLOv1
from data_processing.pascalvoc_yolo import PascalVOCDatasetYOLO
from utils.metrics import intersection_over_union, non_max_suppression, mean_average_precision
from utils.visualization import plot_image, cellboxes_to_boxes, get_bboxes, save_checkpoint, load_checkpoint
from loss.yolo_v1_loss import YOLOv1Loss
from utils.common import Compose, seed_everything
import config.yolov1_config as cfg

# Seed for reproducibility
seed_everything(cfg.SEED)

# transforms for the training data
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

# Train function
def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
        output = model(x)
        loss = loss_fn(output, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    mean_loss = sum(losses)/len(losses)
    print(f"Mean loss was {mean_loss}")


def main():
    # Load model, optimizer and loss function
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    loss_fn = YOLOv1Loss()

    # Load model checkpoint if available
    if cfg.LOAD_MODEL:
        load_checkpoint(torch.load(cfg.LOAD_MODEL_FILE), model, optimizer)

    train_dataset = PascalVOCDatasetYOLO(
        os.path.join(cfg.PROJECT_DIR, "data/PascalVOC_YOLO/8examples.csv"),
        img_dir=cfg.IMG_DIR,
        label_dir=cfg.LABEL_DIR,
        transform=transform
    )
    
    test_dataset = PascalVOCDatasetYOLO(
        os.path.join(cfg.PROJECT_DIR, "data/PascalVOC_YOLO/test.csv"),
        img_dir=cfg.IMG_DIR,
        label_dir=cfg.LABEL_DIR,
        transform=transform
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    for epoch in range(cfg.EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
            train_loader,
            model,
            iou_threshold=0.5,
            threshold=0.4,
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes,
            target_boxes,
            iou_threshold=0.5,
            box_format="midpoint"
        )

        print(f"Train mAP: {mean_avg_prec}")

        train_fn(train_loader, model, optimizer, loss_fn)

if __name__ == "__main__":
    main()