import os
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
from nuimages import NuImages
from nuimages.utils.utils import mask_decode
sys.path.append(os.getenv("TWODOBJECTDETECTION_ROOT"))
from yolov1_nuimages.utils.common import *


class NuImagesDatasetYOLO(torch.utils.data.Dataset):
    """
    Class for a NuImages dataset adapted for YOLOv1.

    Notes:
        - By default, the dataset removes any samples (images) which contain no annotations. These
        cause the training to crash if included. This behavior can be changed by setting the
        'remove_empty' argument when making a NuImagesDataset.
        - Currently, the dataset class skips surface annotations (drivable surfaces and ego vehicle).
        Only object annotations (cars, pedestrians, etc) are included.
        - Any bounding boxes with zero height or width are removed; these cause training to crash
        if left in. The NuImages training set seems to contain one annotation with zero height.

    """

    def __init__(self, nuimages: NuImages, transform=None, remove_empty=True, split_size=7, num_boxes=2, num_classes=25):
        # Check if the nuimages object contains the test set (no annotations)
        if len(nuimages.object_ann) == 0:
            self.has_ann = False
        # Otherwise, dataset is the train or val split
        else:
            self.has_ann = True

        assert type(nuimages) == NuImages
        self.nuimages = nuimages
        self.root_path = Path(nuimages.dataroot)
        self.transform = transform
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        # If training, remove any samples which contain no annotations
        if remove_empty and self.has_ann:
            print("[INFO] Removing samples which contain no annotations...")

            sd_tokens_with_objects = set()
            for object_ann in self.nuimages.object_ann:
                sd_tokens_with_objects.add(object_ann["sample_data_token"])

            self.samples_with_objects = []
            for idx, sample in enumerate(self.nuimages.sample):
                sd_token = sample["key_camera_token"]
                if sd_token in sd_tokens_with_objects:
                    self.samples_with_objects.append(idx)

            print(
                f"[INFO] Done. {len(self.samples_with_objects)} samples remaining out of {len(self.nuimages.sample)}."
            )

        else:
            # Keep all samples if remove_empty set to false
            self.samples_with_objects = [i for i in range(len(self.nuimages.sample))]

        # Create lookup table to convert category name to an int index
        # Speeds up creating annotations
        self._category_name_to_id = {}
        self.category_names = []
        for idx, category in enumerate(nuimages.category):
            self._category_name_to_id[category["name"]] = idx
            self.category_names.append(category["name"])

        # Create lookup table that maps sample data to object annotations
        self.object_anns_dict = {}
        for object_ann in self.nuimages.object_ann:
            object_sd_token = object_ann["sample_data_token"]
            if object_sd_token not in self.object_anns_dict.keys():
                self.object_anns_dict[object_sd_token] = []
            # Remove annotation if bounding box has zero height or width
            # The NuImages training set contains one of these annotations,
            # which crashes the training if it isn't removed
            bbox = object_ann["bbox"]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width > 0 and height > 0:
                self.object_anns_dict[object_sd_token].append(object_ann)

    def __len__(self):
        return len(self.samples_with_objects)

    def __getitem__(self, idx):
        """
        Get an item from the dataset. Returns an image tensor, and a target dict.

        See https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html for the formatting
        of the target dict.

        Parameters
        ----------
        idx: int
            Index of the sample in the dataset.

        Returns
        -------
        image: torch.Tensor
            An RGB training image.
        target: dict
            Dictionary containing the object annotations associated with this image.
        """
        # Get a sample - i.e. an annotated camera image
        sample = self.nuimages.sample[self.samples_with_objects[idx]]
        # Get the associated sample data, representing the image associated with the sample
        sd_token = sample["key_camera_token"]
        sample_data = self.nuimages.get("sample_data", sd_token)
        filename = sample_data["filename"]

        # Read the image file
        image = Image.open(self.root_path / sample_data["filename"]).convert("RGB")

        # If this is the test split (no annotations), just return the image and None for target
        if not self.has_ann:
            return image, None

        # Get the object annotations corresponding to this sample data only
        object_anns = self.object_anns_dict[sd_token]

        # Get bounding boxes
        # Note object_ann['bbox'] gives the bounding box as [x0, y0, x1, y1]
        # Convert to [x, y, w, h] for YOLOv1 as percentages of the image size
        raw_boxes_nuimages = torch.as_tensor([object_ann["bbox"] for object_ann in object_anns], dtype=torch.float32)

        boxes = torch.as_tensor([transform_box_nuimages2yolov1(box, sample_data["width"], sample_data["height"]) for box in raw_boxes_nuimages])

        # Get class labels for each bounding box
        category_tokens = [object_ann["category_token"] for object_ann in object_anns]
        categories = [self.nuimages.get("category", category_token) for category_token in category_tokens]
        class_labels = torch.as_tensor(
            [self._category_name_to_id[categorie["name"]] for categorie in categories],
            dtype=torch.int64,
        )

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Convert box coordinates (for entire image) to cell coordinates (for YOLOv1 grid division).
        label_matrix = torch.zeros((self.S, self.S, self.C + self.B * 5))
        for class_label, box in zip(class_labels, boxes):
            x, y, width, height = box.tolist()
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
