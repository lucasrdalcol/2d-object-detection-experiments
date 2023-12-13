import torch


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (N, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (N, 4)
        box_format (str): midpoint/corners, if boxes (x, y, w, h) or (x1, y1, x2, y2) respectively
    Returns:
        iou (tensor): Intersection over union for all bounding boxes
    """

    # 1. Get the cordinates of bounding boxes
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

    # 2. Get the corrdinates of the intersection boxes
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # 3. Get the area of intersection rectangle
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)  # clamp(min=0) is for the case when they do not intersect

    # 4. Get the area of both boxes
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # 5. Get the area of the union of both boxes
    union = box1_area + box2_area - intersection

    # 6. Calculate the intersection over union
    iou = intersection / (union + 1e-6)  # We add epsilon 1e-6 to avoid division by 0

    return iou


def non_max_suppression(bboxes, iou_threshold, prob_threshold, box_format="corners"):

    assert type(bboxes) == list

    bboxes = [
        box 
        for box in bboxes 
        if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

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