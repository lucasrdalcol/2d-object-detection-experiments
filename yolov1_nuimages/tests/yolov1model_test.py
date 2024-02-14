#!/usr/bin/env python3
import sys
import torch
import os
# sys.path.append("/home/lucasrdalcol/Documents/repositories/2d-object-detection-experiments")
sys.path.append(os.getenv("TWODOBJECTDETECTION_ROOT"))
from yolov1_nuimages.models.yolo_v1 import *


# Test the model to check if everything works fine and the output is what we expect.
def test(split_size=7, num_boxes=2, num_classes=20):
    model = YOLOv1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)
    x = torch.randn((2, 3, 448, 448))
    model_output = model(x)
    print(model_output.shape)

if __name__ == "__main__":
    test()