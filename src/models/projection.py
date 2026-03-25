import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    A non-linear projection head to map the sequence-level global embedding
    into a lower-dimensional space strictly for contrastive alignment.
    """
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        x: (B, 256) - The global representation of the utterance
        returns: (B, 128)
        """
        return self.net(x)