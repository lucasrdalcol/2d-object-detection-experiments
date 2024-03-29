import torch
import os
import sys
import pandas as pd
from PIL import Image
sys.path.append(os.getenv("TWODOBJECTDETECTION_ROOT"))


class PascalVOCDatasetYOLO(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, split_size=7, num_boxes=2, num_classes=20, transform=None, decimation_factor=None):
        self.annotations = pd.read_csv(csv_file)
        if decimation_factor is not None:
            self.annotations = self.annotations.iloc[::decimation_factor, :]
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ] 
                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        filename = self.annotations.iloc[index, 0]
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Convert box coordinates to cell coordinates.
        label_matrix = torch.zeros((self.S, self.S, self.C + self.B * 5))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            # i,j represents the cell row and cell column in the image grid.
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = width * self.S, height * self.S

            # If no object already found for specific cell i,j, use the box and assign the box relative to the cell.
            # Note: This means we restrict to ONE object per cell!
            if label_matrix[i, j, self.C] == 0:
                # Set that there exists an object
                label_matrix[i, j, self.C] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, self.C+1:self.C+5] = box_coordinates
                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix, filename