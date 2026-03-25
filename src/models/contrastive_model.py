import torch
import torch.nn as nn

from src.models.encoder import EMGEncoder
from src.models.decoder import MFCCDecoder
from src.models.projection import ProjectionHead

class EMGContrastiveModel(nn.Module):
    """
    The complete session-invariant model combining the Encoder, Decoder, 
    and the non-linear Projection Head for contrastive learning.
    """
    def __init__(self):
        super().__init__()

        self.encoder = EMGEncoder()
        self.decoder = MFCCDecoder()
        self.projection = ProjectionHead()

    def forward(self, emg, return_latent=False):
        """
        emg: (B, T, 41)

        returns:
            If return_latent=True (used during training):
                mfcc_pred: (B, T, 13)
                z_proj: (B, 128) - The global embedding for Triplet Loss
                
            If return_latent=False (used during inference/evaluation):
                mfcc_pred: (B, T, 13)
        """
        # 1. Get the sequential physiological representation
        z_seq = self.encoder(emg)          # (B, T, 256)

        # 2. Decode into acoustic features (MFCCs) using the full sequence
        mfcc_pred = self.decoder(z_seq)    # (B, T, 13)

        if return_latent:
            # 3. Extract the global sequence-level embedding safely
            # Using mean pooling to capture the entire articulatory sequence
            # regardless of padding or bidirectional overlap.
            z_global = z_seq.mean(dim=1)       # (B, 256)

            # 4. Pass through the non-linear projection head to map to the 
            # contrastive space
            z_proj = self.projection(z_global) # (B, 128)
            
            return mfcc_pred, z_proj

        return mfcc_pred