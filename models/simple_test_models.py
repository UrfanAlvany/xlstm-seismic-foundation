import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class IdentityModel(nn.Module):
    def __init__(self, d_model=None):
        super(IdentityModel, self).__init__()
        self.d_model = d_model

    def forward(self, x, state=None):
        return x, None

    def step(self, x, state=None):
        return x, None


class LinearTestModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lin_1 = nn.Linear(input_size, hidden_size)
        self.lin_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, state):
        x = F.relu(self.lin_1(x))
        x = F.softmax(self.lin_2(x), dim=1)
        return x, state


class ConvNet(nn.Module):
    """
    Simple CNN model to classify MNIST digits. Could be improved with U-Net-like structure / regularization such as dropout layers.
    """

    def __init__(self, in_channels: int = 1, img_size: int = 28):
        super(ConvNet, self).__init__()
        n_features_after_conv = (img_size // 2 // 2 // 2) ** 2 * 128
        self.net = nn.Sequential(
            self.conv_block(in_channels=in_channels,
                            out_channels=32),
            self.conv_block(in_channels=32,
                            out_channels=64),
            self.conv_block(in_channels=64,
                            out_channels=128),
            nn.Flatten(),
            nn.Linear(in_features=n_features_after_conv,
                      out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,
                      out_features=10),
            nn.Softmax(dim=1)
        )

    def conv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

    def forward(self, x, state):
        return self.net(x), state
