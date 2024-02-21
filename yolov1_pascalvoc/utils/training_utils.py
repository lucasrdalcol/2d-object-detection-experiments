from tqdm import tqdm
import torch
import sys
import os

import wandb

sys.path.append(os.getenv("TWODOBJECTDETECTION_ROOT"))
from yolov1_pascalvoc.utils.metrics import *


# Train function
def train_epoch(train_dataloader, model, optimizer, loss_fn, device="cuda"):
    train_dataloader_loop = tqdm(train_dataloader)
    model.train()  # set model to training mode
    losses = []

    # Iterate over the training data
    for batch_idx, (inputs_x, labels_y, _) in enumerate(train_dataloader_loop):
        inputs_x, labels_y = inputs_x.to(device), labels_y.to(
            device
        )  # Move data to device (GPU if available)

        # forward pass: Feed inputs to the model and compute loss.
        predictions = model(inputs_x)  # Feed inputs to the model to get predictions
        loss = loss_fn(predictions, labels_y)  # calculate loss
        losses.append(loss.item())

        # backward pass: compute gradient of the loss with respect to model parameters
        optimizer.zero_grad()  # zero out the gradients from the previous step
        loss.backward()  # backpropagate the loss
        optimizer.step()  # perform a single optimization step (model parameter update)

        train_dataloader_loop.set_postfix(loss=loss.item())  # update progress bar
        
        # Clear memory
        del loss, predictions
        torch.cuda.empty_cache()  # Clear GPU memory

    # Compute mean loss over all batches of the training data
    mean_loss = sum(losses) / len(losses)
    print(f"Train mean loss: {mean_loss}")

    return mean_loss


def validate_epoch(
    val_dataloader,
    model,
    loss_fn,
    iou_threshold,
    prob_threshold,
    box_format="midpoint",
    device="cuda",
):
    """
    Perform validation for one epoch.

    Args:
        val_dataloader (torch.utils.data.DataLoader): The validation data loader.
        model (torch.nn.Module): The model to be evaluated.
        loss_fn (callable): The loss function used for calculating the loss.
        iou_threshold (float): The IoU threshold for non-maximum suppression.
        prob_threshold (float): The probability threshold for filtering predictions.
        box_format (str, optional): The format of the bounding boxes. Defaults to "midpoint".
        device (str, optional): The device to be used for computation. Defaults to "cuda".

    Returns:
        tuple: A tuple containing the mean loss and mean average precision.
    """
    val_dataloader_loop = tqdm(val_dataloader)
    model.eval()  # set model to evaluation mode
    losses = []
    all_pred_boxes = []
    all_true_boxes = []
    train_idx = 0

    # Iterate over the validation data
    with torch.no_grad():  # disable gradient calculation
        for batch_idx, (inputs_x, labels_y, _) in enumerate(val_dataloader_loop):
            # forward pass: Feed inputs to the model and compute loss.
            # No need to compute gradients in validation phase (backpropagation)
            inputs_x, labels_y = inputs_x.to(device), labels_y.to(
                device
            )  # Move data to device (GPU if available)
            predictions = model(inputs_x)  # Feed inputs to the model to get predictions
            loss = loss_fn(predictions, labels_y)  # calculate loss
            losses.append(loss.item())

            val_dataloader_loop.set_postfix(loss=loss.item())  # update progress bar

            batch_size = inputs_x.shape[0]
            true_bboxes = cellboxes_to_boxes(labels_y)
            pred_bboxes = cellboxes_to_boxes(predictions)

            for idx in range(batch_size):
                nms_boxes = non_max_suppression(
                    pred_bboxes[idx],
                    iou_threshold=iou_threshold,
                    prob_threshold=prob_threshold,
                    box_format=box_format,
                )

                # if batch_idx == 0 and idx == 0:
                #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
                #    print(nms_boxes)

                for nms_box in nms_boxes:
                    all_pred_boxes.append([train_idx] + nms_box)

                for true_box in true_bboxes[idx]:
                    # many will get converted to 0 pred
                    if true_box[1] > prob_threshold:
                        all_true_boxes.append([train_idx] + true_box)

                train_idx += 1

            # Clear memory
            del loss, predictions
            torch.cuda.empty_cache()  # Clear GPU memory

    # Compute mean loss over all batches of the validation data
    mean_loss = sum(losses) / len(losses)
    print(f"Val mean loss: {mean_loss}")

    # Compute mean average precision
    mean_avg_prec = mean_average_precision(
        all_pred_boxes,
        all_true_boxes,
        iou_threshold=iou_threshold,
        box_format=box_format,
    )

    return mean_loss, mean_avg_prec


def get_bboxes(
    dataloader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
    progress_bar=False
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
    if progress_bar:
        dataloader_loop = tqdm(dataloader)

    for batch_idx, (inputs_x, labels_y, _) in enumerate(dataloader_loop if progress_bar else dataloader):
        inputs_x = inputs_x.to(device)
        labels_y = labels_y.to(device)

        with torch.no_grad():
            predictions = model(inputs_x)

        batch_size = inputs_x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels_y)
        pred_bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                pred_bboxes[idx],
                iou_threshold=iou_threshold,
                prob_threshold=threshold,
                box_format=box_format,
            )

            # if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for true_box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if true_box[1] > threshold:
                    all_true_boxes.append([train_idx] + true_box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


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
