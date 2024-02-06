import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

import wandb

sys.path.append(os.getenv("TWODOBJECTDETECTION_ROOT"))
from utils.metrics import *


def plot_comparison_image(image, filename, pred_boxes, true_boxes):
    """
    Plots predicted and true bounding boxes on the same image side by side.

    Args:
        image: The input image.
        boxes: A list of bounding boxes.

    Returns:
        None
    """
    # Be sure to close plt figure
    plt.close()
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(filename)
    # Display the image
    ax1.imshow(im)
    ax1.set_title("Predicted bboxes")
    ax2.imshow(im)
    ax2.set_title("True bboxes")

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create rectangles for predicted bounding boxes
    for pred_box in pred_boxes:
        pred_box = pred_box[2:]
        assert len(pred_box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = pred_box[0] - pred_box[2] / 2
        upper_left_y = pred_box[1] - pred_box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            pred_box[2] * width,
            pred_box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax1.add_patch(rect)

    # Create rectangles for true bounding boxes
    for true_box in true_boxes:
        true_box = true_box[2:]
        assert len(true_box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = true_box[0] - true_box[2] / 2
        upper_left_y = true_box[1] - true_box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            true_box[2] * width,
            true_box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax2.add_patch(rect)

    return plt


def wandb_bounding_boxes(raw_image, filename, bboxes, class_id_to_label):
    all_boxes = []
    # plot each bounding box for this image
    for box in bboxes:
        if box[1] == 0.0:
            continue
        # get coordinates and labels
        box_data = {
            "position": {
                "middle": [box[2], box[3]],
                "width": box[4],
                "height": box[5],
            },
            "class_id": int(box[0]),
            # optionally caption each box with its class and score
            "box_caption": "%s (%.1f)" % (class_id_to_label[box[0]], box[1] * 100),
            "scores": {"score": box[1] * 100},
        }
        all_boxes.append(box_data)

    # log to wandb: raw image, predictions, and dictionary of class labels for each class id
    box_image = wandb.Image(
        raw_image,
        caption=filename,
        boxes={
            "predictions": {"box_data": all_boxes, "class_labels": class_id_to_label}
        },
    )
    return box_image
