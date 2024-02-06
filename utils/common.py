import random
import numpy as np
import torch
import sys
import os

sys.path.append(os.getenv("TWODOBJECTDETECTION_ROOT"))


class Compose(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        """
        Args:
            transforms: List of transforms to compose.
        """
        self.transforms = transforms

    def __call__(self, img, bboxes, transform_bbox=False):
        """
        Args:
            img: PIL image.
            boxes: Bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4).

        Returns:
            PIL image: Transformed PIL image.
            tensor: Transformed bounding box of dimensions (n_objects, 4).
        """
        for t in self.transforms:
            if transform_bbox:
                img, bboxes = t(img, bboxes)
            else:
                img, bboxes = t(img), bboxes

        return img, bboxes


def seed_everything(seed=123):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int, optional): Number of the seed. Defaults to 123.

    Returns:
        None
    """
    # Set seed for python
    random.seed(seed)

    # Set seed for python with numpy
    np.random.seed(seed)

    # Set seed for pytorch
    torch.manual_seed(seed)

    # Set seed for CUDA
    torch.cuda.manual_seed(seed)

    # Set seed for CUDA with new generator
    torch.cuda.manual_seed_all(seed)

    # Set seed for all devices (GPU and CPU)
    torch.backends.cudnn.deterministic = True

    # Set seed for all devices (GPU and CPU) - faster but not deterministic
    torch.backends.cudnn.benchmark = True


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
