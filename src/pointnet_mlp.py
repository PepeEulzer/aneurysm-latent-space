import torch
import torch.nn as nn
import numpy as np

# github: https://github.com/MaciejZamorski/3d-AAE/blob/master/models/aae.py
# Paper: https://wendy-xiao.github.io/files/pointer_cloud.pdf
# Alternatives: https://github.com/charlesq34/pointnet-autoencoder/tree/master


class PointNetMLP(nn.Module):
    def __init__(self, z_size, use_bias, num_classes):
        super().__init__()

        self.z_size = z_size
        self.use_bias = use_bias
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=self.num_classes, bias=self.use_bias),
        ) 

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, bias=self.use_bias),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.z_size, bias=True),
        )

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        z = self.fc(output2)
        return z


class PointNetMLPNorm(nn.Module):
    def __init__(self, z_size, num_classes, dropout=0.5): # Add num_classes
        super().__init__()
        self.z_size = z_size
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1, bias=True),
            nn.BatchNorm1d(512), # Added BatchNorm
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),  
            nn.Linear(256, self.z_size, bias=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.z_size, self.num_classes, bias=True), # Added Bias, initialized num_classes
        )

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        z = self.fc(output2)
        return z