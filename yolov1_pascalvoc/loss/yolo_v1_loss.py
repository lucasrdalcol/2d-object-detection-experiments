import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.getenv("TWODOBJECTDETECTION_ROOT"))
from yolov1_pascalvoc.utils.metrics import *

class YOLOv1Loss(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20):
        super(YOLOv1Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        # Lambdas are used to weight the loss of specific components.
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5) # tensor shape (N, S, S, C + B * 5)
        iou_b1 = intersection_over_union(predictions[..., self.C+1:self.C+5], target[..., self.C+1:self.C+5])
        iou_b2 = intersection_over_union(predictions[..., self.C+6:self.C+10], target[..., self.C+1:self.C+5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) 
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., self.C:self.C+1] # Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        box_predictions = exists_box * (
            (best_box * predictions[..., self.C+6:self.C+10]
            + (1 - best_box) * predictions[..., self.C+1:self.C+5])
        )
        box_targets = exists_box * target[..., self.C+1:self.C+5]

        # Take sqrt of width, height of boxes to ensure that they are not negative.
        # Not necessary for x1 and y1 midpoints, since they are already between 0 and 1.
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # ======================== #
        #   FOR OBJECT LOSS        #
        # ======================== #
        pred_box = (
            best_box * predictions[..., self.C+5:self.C+6]
            + (1 - best_box) * predictions[..., self.C:self.C+1]
        )

        # (N, S, S, 1) -> (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.C:self.C+1])
        )

        # ======================== #
        #   FOR NO OBJECT LOSS     #
        # ======================== #
        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C:self.C+1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1)
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C+5:self.C+6], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1)
        )

        # ======================== #
        #   FOR CLASS LOSS         #
        # ======================== #
        # (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        )

        # ======================== #
        #   COMPLETE LOSS          #
        # ======================== #
        loss = (
            self.lambda_coord * box_loss  # First two rows of loss in paper.
            + object_loss # Third row of loss in paper.
            + self.lambda_noobj * no_object_loss # Fourth row of loss in paper.
            + class_loss # Fifth row of loss in paper.
        )

        return loss