import torch
import torch.nn as nn


class EMGEncoder(nn.Module):
    def __init__(self, input_dim=41, hidden_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        # x: (B, T, 41)

        x = x.transpose(1, 2)   # (B, 41, T)
        x = self.conv(x)        # (B, 128, T)
        x = x.transpose(1, 2)   # (B, T, 128)

        x, _ = self.lstm(x)     # (B, T, 256)

        return x