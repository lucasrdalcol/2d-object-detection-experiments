import importlib
import torch
import torch.nn as nn
import torchvision


class YOLOv1PreTrainedResnet50(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YOLOv1PreTrainedResnet50, self).__init__()
        self.in_channels = in_channels

        # Load pre-trained ResNet model
        resnet50 = torchvision.models.resnet50(weights="DEFAULT")
        # Remove the fully connected layer and the last pooling layer
        layers = list(resnet50.children())[:-2]
        self.cnn = nn.Sequential(*layers)
        # Freeze the parameters of the pre-trained ResNet
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x

    # This function creates the fully connected layers
    # The functions starting with _ are internal functions only used by the class.
    def _create_fcs(self, split_size, num_boxes, num_classes, fcl_size, dropout):
        S, B, C = split_size, num_boxes, num_classes
        # Way of doing it with nn.Sequential:
        fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                7 * 7 * 2048, fcl_size
            ),  # In the original paper fcl_size should be 4096. The input size is 7x7x2048 due to the last conv layer of the ResNet50
            nn.Dropout(dropout),  # In the original paper this should be 0.5
            nn.LeakyReLU(0.1),
            nn.Linear(
                fcl_size, S * S * (C + B * 5)
            ),  # reshape afterwards to shape (S, S, 35) where C + B * 5 = 35, for the loss function
        )

        return fcs
