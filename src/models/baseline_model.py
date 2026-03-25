import torch
import torch.nn as nn

from src.models.encoder import EMGEncoder
from src.models.decoder import MFCCDecoder


class EMGBaselineModel(nn.Module):
    """
    Simple encoder-decoder model:
    EMG → MFCC
    """

    def __init__(self):
        super().__init__()

        self.encoder = EMGEncoder()
        self.decoder = MFCCDecoder()

    def forward(self, emg):
        """
        emg: (B, T, 41)

        returns:
            mfcc_pred: (B, T, 13)
        """

        z = self.encoder(emg)        # (B, T, 256)
        mfcc_pred = self.decoder(z)  # (B, T, 13)

        return mfcc_pred