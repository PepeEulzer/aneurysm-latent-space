import torch
import torch.nn as nn
import numpy as np

# github: https://github.com/MaciejZamorski/3d-AAE/blob/master/models/aae.py
# Paper: https://wendy-xiao.github.io/files/pointer_cloud.pdf
# Alternatives: https://github.com/charlesq34/pointnet-autoencoder/tree/master


class VAEDecoder(nn.Module):
    def __init__(self, z_size, use_bias, out_dim=716):
        super().__init__()

        self.z_size = z_size
        self.use_bias = use_bias 
        self.out_dim = out_dim

        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=self.out_dim * 3, bias=self.use_bias),
        )

    def forward(self, input):
        z, mu, logvar = input
        output = self.model(z.squeeze())
        output = output.view(-1, 3, self.out_dim)
        return output

class VAEEncoder(nn.Module):
    def __init__(self, z_size, use_bias):
        super().__init__()

        self.z_size = z_size
        self.use_bias = use_bias
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.num_classes = 2
        self.classifier = nn.Sequential(
            #nn.Linear(in_features=self.z_size, out_features=512, bias=self.use_bias),
            #nn.ReLU(inplace=True),

            nn.Linear(in_features=self.z_size, out_features=self.num_classes, bias=self.use_bias),
        ) 
        self.embedding_layer = nn.Embedding(self.num_classes, self.z_size)

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
            nn.ReLU(inplace=True)
        )

        self.mu_layer = nn.Linear(256, self.z_size, bias=True)
        self.std_layer = nn.Linear(256, self.z_size, bias=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        logit = self.fc(output2)
        mu = self.mu_layer(logit)
        logvar = self.std_layer(logit)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar