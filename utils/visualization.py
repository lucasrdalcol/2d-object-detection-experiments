import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import non_max_suppression

def plot_image(image, boxes):
    """
    Plots predicted bounding boxes on the image.

    Args:
        image: The input image.
        boxes: A list of bounding boxes.

    Returns:
        None
    """
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    """
    Get bounding boxes for object detection predictions.

    Args:
        loader (torch.utils.data.DataLoader): Data loader for the dataset.
        model (torch.nn.Module): Object detection model.
        iou_threshold (float): IoU threshold for non-maximum suppression.
        threshold (float): Confidence threshold for object detection.
        pred_format (str, optional): Format of the predicted bounding boxes. Defaults to "cells".
        box_format (str, optional): Format of the bounding boxes. Defaults to "midpoint".
        device (str, optional): Device to use for computation. Defaults to "cuda".

    Returns:
        tuple: A tuple containing two lists. The first list contains the predicted bounding boxes,
               and the second list contains the true bounding boxes.
    """
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                prob_threshold=threshold,
                box_format=box_format,
            )


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios.

    Args:
        predictions (torch.Tensor): The predictions tensor of shape (batch_size, 7, 7, 30).
        S (int, optional): The image split size. Defaults to 7.

    Returns:
        torch.Tensor: The converted predictions tensor of shape (batch_size, 7, 7, 5).

    Note:
        This function converts the bounding boxes from relative cell ratios to entire image ratios.
        It uses the YOLO output format, where each cell predicts 2 bounding boxes and 20 classes.
        The converted predictions tensor has the format (class, confidence, x, y, w, h).

    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    """
    Convert cellboxes to bounding boxes.

    Args:
        out (Tensor): The output tensor containing cellboxes.
        S (int): The number of cells in each dimension.

    Returns:
        List[List[List[float]]]: A list of bounding boxes for each example.
    """
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Save the model checkpoint to a file.

    Args:
        state (dict): The state dictionary containing the model's state.
        filename (str): The name of the file to save the checkpoint to. Default is "my_checkpoint.pth.tar".
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    """
    Loads the model and optimizer states from a checkpoint.

    Args:
        checkpoint (dict): The checkpoint containing the model and optimizer states.
        model (nn.Module): The model to load the state_dict into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state_dict into.
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])