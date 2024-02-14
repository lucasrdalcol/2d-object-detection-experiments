import torch
import torch.nn as nn


architecture_config = [
    # conv config tuple: (kernel_size, number of filters, stride, padding)
    (7, 64, 2, 3),
    "M",  # MaxPool2d(kernel_size=2, stride=2)
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # list: [(kernel_size, number of filters, stride, padding), (...), number of repetitions)]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
    # does not include the fully connected layers
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        # First way of implementing it:
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

        # # Second way of implementing it:
        # self.model = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
        #     nn.BatchNorm2d(out_channels),
        #     nn.LeakyReLU(0.1),
        # )

    def forward(self, x):
        # First way of implementing it:
        return self.leaky_relu(self.batch_norm(self.conv(x)))
        # # Second way of implementing it:
        # return self.model(x)


class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YOLOv1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x

    # This function creates the convolutional layers
    # The functions starting with _ are internal functions only used by the class.
    # Cleaver way of creating blocks of layers, such as conv blocks and fc blocks.
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels,
                        out_channels=x[1],
                        kernel_size=x[0],
                        stride=x[2],
                        padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) == list:
                conv1 = x[0]  # tuple
                conv2 = x[1]  # tuple
                num_repeats = x[2]  # integer

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            out_channels=conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]

                    layers += [
                        CNNBlock(
                            in_channels=conv1[1],
                            out_channels=conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]

                    in_channels = conv2[1]

        return nn.Sequential(*layers)  # * unpacks the list to give to the function

    # This function creates the fully connected layers
    # The functions starting with _ are internal functions only used by the class.
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        # Way of doing it with nn.Sequential:
        fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),  # In the original paper this should be 4096
            nn.Dropout(0.0),  # In the original paper this should be 0.5
            nn.LeakyReLU(0.1),
            nn.Linear(
                496, S * S * (C + B * 5)
            ),  # reshape afterwards to shape (S, S, 30) where C + B * 5 = 35, for the loss function
        )

        return fcs
