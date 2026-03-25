import os
import json
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random


class StandardEMGDataset(Dataset):
    def __init__(self, root_dir, speaker="Spk1", split="train", blocks=None):
        """
        Args:
            root_dir (str): Path to CSL-EMG_Array dataset folder.
            speaker (str): Speaker ID (e.g., "Spk1").
            split (str): "train", "dev", or "eval".
            blocks (list): Which blocks to pull from. If None, it pulls from all applicable blocks.
        """
        self.root_dir = root_dir
        self.speaker = speaker
        self.split = split
        
        # Default block mapping based on the corpus design
        if blocks is None:
            if split == "train":
                self.blocks = ["Block1-Initial", "Block2-Adapt1", "Block4-Adapt2", "Block6-Adapt3"]
            else: # dev or eval
                self.blocks = [ "Block3-Eval1", "Block5-Eval2", "Block7-Eval3"]
        else:
            self.blocks = blocks
            
        self.samples = [] # Stores (block_folder, idx)
        
        self._build_index()

    def _build_index(self):
        for block in self.blocks:
            block_folder = f"{self.speaker}_{block}"
            json_path = os.path.join(self.root_dir, block_folder, "recordingLog.json")
            
            if not os.path.exists(json_path):
                continue
                
            with open(json_path, 'r') as f:
                log_data = json.load(f)
                
            # Safely check if the requested split exists in this block's JSON
            if "uttSets" in log_data and self.split in log_data["uttSets"]:
                # log_data["uttSets"]["train"] -> ["0001", "0002", ...]
                for idx in log_data["uttSets"][self.split]:
                    self.samples.append((block_folder, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        block_folder, idx = self.samples[index]
        base_path = os.path.join(self.root_dir, block_folder, str(idx), f"{block_folder}_{idx}")
        
        # 1. Directly load the pre-calculated, pre-aligned arrays
        emg_aligned = np.load(f"{base_path}_emg_proc.npy")
        mfcc_extracted = np.load(f"{base_path}_mfcc.npy")
        
        # 2. Convert to tensors and return
        return torch.tensor(emg_aligned, dtype=torch.float32), torch.tensor(mfcc_extracted, dtype=torch.float32)
    

def baseline_collate_fn(batch):
    """Pads varying-length EMG and MFCC sequences to the max length in the batch."""
    emg_list, mfcc_list = [], []
    
    for emg, mfcc in batch:
        emg_list.append(emg)
        mfcc_list.append(mfcc)
        
    emg_padded = pad_sequence(emg_list, batch_first=True)
    mfcc_padded = pad_sequence(mfcc_list, batch_first=True)
    
    # We also return the original lengths so we can optionally mask the padding during loss calculation
    lengths = torch.tensor([len(emg) for emg in emg_list])
    
    return emg_padded, mfcc_padded, lengths

