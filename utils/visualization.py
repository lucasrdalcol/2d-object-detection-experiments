import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import *


def plot_comparison_image(image, img_name, pred_boxes, true_boxes, save=False, save_path=None):
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
    fig.suptitle(img_name)
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
