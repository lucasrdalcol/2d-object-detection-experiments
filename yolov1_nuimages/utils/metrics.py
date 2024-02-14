import torch
from collections import Counter
import sys
import os
sys.path.append(os.getenv("TWODOBJECTDETECTION_ROOT"))

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (N, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (N, 4)
        box_format (str): midpoint/corners, if boxes (x, y, w, h) or (x1, y1, x2, y2) respectively
    Returns:
        tensor: Intersection over union for all bounding boxes
    """

    # Get the cordinates of bounding boxes
    # boxes_pred and boxes_labels shapes are (N, 4), where N is the number of bboxes
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - (boxes_preds[..., 2:3] / 2) # box1 top left x coordinate. shape (N, 1)
        box1_y1 = boxes_preds[..., 1:2] - (boxes_preds[..., 3:4] / 2) # box1 top left y coordinate. shape (N, 1)
        box1_x2 = boxes_preds[..., 0:1] + (boxes_preds[..., 2:3] / 2) # box1 bottom right x coordinate. shape (N, 1)
        box1_y2 = boxes_preds[..., 1:2] + (boxes_preds[..., 3:4] / 2) # box1 bottom right y coordinate. shape (N, 1)
        box2_x1 = boxes_labels[..., 0:1] - (boxes_labels[..., 2:3] / 2) # box2 top left x coordinate. shape (N, 1)
        box2_y1 = boxes_labels[..., 1:2] - (boxes_labels[..., 3:4] / 2) # box2 top left y coordinate. shape (N, 1)
        box2_x2 = boxes_labels[..., 0:1] + (boxes_labels[..., 2:3] / 2) # box2 bottom right x coordinate. shape (N, 1)
        box2_y2 = boxes_labels[..., 1:2] + (boxes_labels[..., 3:4] / 2) # box2 bottom right y coordinate. shape (N, 1)
    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]  # box1 top left x coordinate. shape (N, 1)
        box1_y1 = boxes_preds[..., 1:2]  # box1 top left y coordinate. shape (N, 1)
        box1_x2 = boxes_preds[..., 2:3]  # box1 bottom right x coordinate. shape (N, 1)
        box1_y2 = boxes_preds[..., 3:4]  # box1 bottom right y coordinate. shape (N, 1)
        box2_x1 = boxes_labels[..., 0:1]  # box2 top left x coordinate. shape (N, 1)
        box2_y1 = boxes_labels[..., 1:2]  # box2 top left y coordinate. shape (N, 1)
        box2_x2 = boxes_labels[..., 2:3]  # box2 bottom right x coordinate. shape (N, 1)
        box2_y2 = boxes_labels[..., 3:4]  # box2 bottom right y coordinate. shape (N, 1)

    # Get the cordinates of the intersection boxes
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Get the area of intersection rectangle
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)  # clamp(min=0) is for the case when they do not intersect

    # Get the area of both boxes
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # Get the area of the union of both boxes
    union = box1_area + box2_area - intersection

    # Calculate the intersection over union
    iou = intersection / (union + 1e-6)  # We add epsilon 1e-6 to avoid division by 0

    return iou


def non_max_suppression(bboxes, iou_threshold, prob_threshold, box_format="corners"):
    """
    Applies non-maximum suppression to a list of bounding boxes.

    Args:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): IoU threshold where predicted bboxes is correct
        prob_threshold (float): Probability threshold for filtering bboxes based on their confidence score.
        box_format (str, optional): Format of the bounding boxes, "midpoint" or "corners" used to specify bboxes. Defaults to "corners".

    Returns:
        list: List of bounding boxes after non-maximum suppression.

    """
    assert type(bboxes) == list

    # Create a new list containing only the bboxes that have a confidence score higher than the threshold and sort them by confidence score
    bboxes = [
        box 
        for box in bboxes 
        if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    # Apply non-maximum suppression algorithm
    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format=box_format) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    """
    Calculates mean average precision for a specific IoU threshold.

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """
    # List storing all AP for respective classes
    average_precisions_per_class = []
    epsilon = 1e-6

    # Go through all classes
    for c in range(num_classes):
        detections_per_class = []
        gt_per_class = []

        # Go through all predictions and targets, and only add the ones that belong to the current class c.
        for detection in pred_boxes:
            if detection[1] == c:
                detections_per_class.append(detection)
        for gt in true_boxes:
            if gt[1] == c:
                gt_per_class.append(gt)

        # Go through all the amount of ground truth bboxes per image and create a tensor with zeros
        # amount_gt_per_img = {0: 3, 1:5} for image 0 with 3 bboxes and image 1 with 5 bboxes
        amount_gt_per_img = Counter([gt[0] for gt in gt_per_class]) # amount of ground truth bboxes per image
        for img_idx, amount_gt in amount_gt_per_img.items():
            # amount_gt_per_img = {0: torch.tensor([0, 0, 0]), 1: torch.tensor([0, 0, 0, 0, 0])}
            amount_gt_per_img[img_idx] = torch.zeros(amount_gt)

        # Create tensors for true positives and false positives for current class c
        detections_per_class.sort(key=lambda x: x[2], reverse=True) # descendent sort by confidence score
        true_positives_per_class = torch.zeros((len(detections_per_class)))
        false_positives_per_class = torch.zeros((len(detections_per_class)))
        total_gt_per_class = len(gt_per_class)

        # If no true boxes exists for this class, then we can safely skip
        if total_gt_per_class == 0:
            continue

        # Go through all detections in detections_per_class and check if they are true positives or false positives
        for detection_idx, detection in enumerate(detections_per_class):
            gt_per_img = [gt for gt in gt_per_class if gt[0] == detection[0]]

            best_iou = 0
            # Compare only the current detection with the ground truth bboxes that belong to the same image
            for gt_idx, gt in enumerate(gt_per_img):
                iou = intersection_over_union(torch.tensor(detection[3:]), 
                                              torch.tensor(gt[3:]), 
                                              box_format=box_format)

                # When comparing the detection with the ground truth bboxes for that image, 
                # we only consider the one with the highest IoU
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # After finding the correspondent gt bbox for that detection, 
            # we check if it is higher than the iou threshold and if it is, 
            # we check if it has already been detected. 
            # If not, it is a true positive detection, otherwise it is a false positive.
            if best_iou > iou_threshold:
                if amount_gt_per_img[detection[0]][best_gt_idx] == 0:
                    true_positives_per_class[detection_idx] = 1
                    amount_gt_per_img[detection[0]][best_gt_idx] = 1 # mark that this target bbox has already been detected
                else:
                    # If the target bbox has already been detected by a bbox with highest confidence, it is a false positive
                    false_positives_per_class[detection_idx] = 1
            else:
                # If the IoU is lower than the threshold, it is a false positive
                false_positives_per_class[detection_idx] = 1 

        # Calculate precision and recall for current class c
        true_positives_per_class_cumsum = torch.cumsum(true_positives_per_class, dim=0) # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        false_positives_per_class_cumsum = torch.cumsum(false_positives_per_class, dim=0) # [0, 0, 1, 0, 1] -> [0, 0, 1, 1, 2]
        recalls_per_class = true_positives_per_class_cumsum / (total_gt_per_class + epsilon) # [1, 2, 2, 3, 3] / 3 = [0.33, 0.66, 0.66, 1, 1]
        precisions_per_class = true_positives_per_class_cumsum / (true_positives_per_class_cumsum + false_positives_per_class_cumsum + epsilon) # [1, 2, 2, 3, 3] / [1, 2, 3, 4, 5] = [1, 1, 0.66, 0.75, 0.6]
        recalls_per_class = torch.cat((torch.tensor([0]), recalls_per_class)) # [0, 0.33, 0.66, 0.66, 1, 1]
        precisions_per_class = torch.cat((torch.tensor([1]), precisions_per_class)) # [1, 1, 1, 0.66, 0.75, 0.6]

        average_precisions_per_class.append(torch.trapz(precisions_per_class, recalls_per_class)) # area under precision-recall curve, numerical integration

    # Calculate mean average precision across all classes
    result_map = sum(average_precisions_per_class) / len(average_precisions_per_class)
    return result_map