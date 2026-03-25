import torch
import torch.nn as nn


class MFCCDecoder(nn.Module):
    def __init__(self, input_dim=256, output_dim=13):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # x: (B, T, 256)
        return self.net(x)  # (B, T, 13)