import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletContrastiveLoss(nn.Module):
    """
    Triplet loss with mandatory L2 Normalization for contrastive embedding stability.
    Forces Anchor and Positive to be closer than Anchor and Negative by at least `margin`.
    """
    def __init__(self, margin=1.0, normalize_embeddings=True):
        super().__init__()
        self.margin = margin
        self.normalize_embeddings = normalize_embeddings
        
        # p=2 means we use standard Euclidean distance in the embedding space
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, z_anchor, z_pos, z_neg):
        """
        Inputs should be the output of the ProjectionHead: (B, 128)
        """
        if self.normalize_embeddings:
            # L2 Normalize to prevent the network from exploding the embedding magnitudes
            z_anchor = F.normalize(z_anchor, p=2, dim=1)
            z_pos    = F.normalize(z_pos, p=2, dim=1)
            z_neg    = F.normalize(z_neg, p=2, dim=1)

        return self.triplet_loss(z_anchor, z_pos, z_neg)