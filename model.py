import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return x + self.conv(x)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, r=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // r, 1, 1, 0),
            nn.BatchNorm2d(out_channels // r),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels // r, out_channels // r, 3, 1, 1),
            nn.BatchNorm2d(out_channels // r),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels // r, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return x + self.conv(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Net(nn.Module):
    def __init__(self, in_channels, latent_dim, dims=[64, 128, 256, 512]):
        super().__init__()

        self.depth = len(dims)
        self.hidden_dim = dims[-1]

        self.mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.LeakyReLU(),
            nn.Linear(latent_dim * 4, latent_dim)
        )

        # encoder
        for i in range(len(dims)):
            setattr(self,
                'e{}'.format(i + 1),
                nn.Sequential(
                    nn.Conv2d(in_channels, dims[i], 2, 2, 0),
                    nn.BatchNorm2d(dims[i]),
                    nn.LeakyReLU(),
                    nn.Conv2d(dims[i], dims[i], 3, 1, 1),
                    nn.BatchNorm2d(dims[i]),
                    nn.LeakyReLU()
                )
            )
            in_channels = dims[i]

        self.encoder_fc = nn.Linear(dims[-1] * 4, latent_dim)

        # decoder
        self.decoder_fc = nn.Linear(latent_dim * 2, dims[-1] * 4)
        self.decoder_fc_post = nn.Sequential(
            nn.BatchNorm2d(dims[-1]),
            nn.LeakyReLU()
        )

        dims.reverse()
        for i in range(len(dims) - 1):
            setattr(self,
                'd{}'.format(i + 1),
                nn.Sequential(
                    nn.ConvTranspose2d(dims[i], dims[i + 1], 2, 2, 0, groups=max(1, dims[i] // 128)),
                    nn.BatchNorm2d(dims[i + 1]),
                    nn.LeakyReLU(),
                    BottleneckBlock(dims[i + 1], dims[i + 1], r=max(1, dims[i + 1] // 64)),
                    BottleneckBlock(dims[i + 1], dims[i + 1], r=max(1, dims[i + 1] // 64))
                )
            )

        self.output = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], dims[-1] // 2, 2, 2, 0),
            nn.BatchNorm2d(dims[-1] // 2),
            nn.LeakyReLU(),
            ResBlock(dims[-1] // 2, dims[-1] // 2),
            ResBlock(dims[-1] // 2, dims[-1] // 2),
            nn.Conv2d(dims[-1] // 2, 3, 3, 1, 1)
        )

    def forward(self, x, m, r, train=True):
        # encoder
        for i in range(self.depth):
            if i > 0:
                x = x * m[i - 1].to(device).unsqueeze(2).unsqueeze(3)  # channel dropout
            conv = getattr(self, 'e{}'.format(i + 1))
            x = conv(x)

        x = self.encoder_fc(torch.flatten(x, start_dim=1))
        r = self.mlp(r)

        # decoder
        x = self.decoder_fc(torch.cat([x, r], 1))
        x = self.decoder_fc_post(x.view(-1, self.hidden_dim, 2, 2))

        for i in range(self.depth - 1):
            conv = getattr(self, 'd{}'.format(i + 1))
            x = conv(x)

        return self.output(x)
