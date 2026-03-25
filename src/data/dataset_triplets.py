import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class EMGContrastiveDataset(Dataset):
    def __init__(self, processed_dir="CSL-EMG_Processed", speaker="Spk1", train_blocks=["Block1-Initial", "Block2-Adapt1", "Block4-Adapt2", "Block6-Adapt3"]):
        """
        Args:
            processed_dir (str): Path to the PREPROCESSED data folder.
            speaker (str): Speaker ID (e.g., "Spk1").
            train_blocks (list): Blocks to use for training.
        """
        self.root_dir = processed_dir
        self.speaker = speaker
        self.train_blocks = train_blocks
        
        self.utterances = [] 
        self.sentence_map = {} 
        
        self._build_index()

    def _build_index(self):
        """Reads JSON logs to map texts to their respective block/IDs."""
        for block in self.train_blocks:
            block_folder = f"{self.speaker}_{block}"
            json_path = os.path.join(self.root_dir, block_folder, "recordingLog.json")
            
            if not os.path.exists(json_path):
                print(f"Warning: Could not find {json_path}")
                continue
                
            with open(json_path, 'r') as f:
                log_data = json.load(f)
                
            if "promptList" in log_data:
                for idx, text in log_data["promptList"].items():
                    self.utterances.append((block_folder, idx, text))
                    
                    if text not in self.sentence_map:
                        self.sentence_map[text] = []
                    self.sentence_map[text].append((block_folder, idx))

    def _load_tensors(self, block_folder, idx):
        """Simply loads the preprocessed arrays from disk."""
        base_path = os.path.join(self.root_dir, block_folder, str(idx), f"{block_folder}_{idx}")
        
        # Load the already-normalized and aligned arrays!
        emg_proc = np.load(f"{base_path}_emg_proc.npy")
        mfcc_proc = np.load(f"{base_path}_mfcc.npy")
        
        return torch.tensor(emg_proc, dtype=torch.float32), torch.tensor(mfcc_proc, dtype=torch.float32)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index):
        anchor_block, anchor_idx, text = self.utterances[index]
        
        # 1. Load Anchor
        emg_a, mfcc_a = self._load_tensors(anchor_block, anchor_idx)
        
        # 2. Mine Positive (Same text, DIFFERENT block)
        available_positives = [x for x in self.sentence_map[text] if x[0] != anchor_block]
        
        if available_positives:
            pos_block, pos_idx = random.choice(available_positives)
            emg_p, _ = self._load_tensors(pos_block, pos_idx)
        else:
            # Fallback for unique Block 1 sentences: clone and add mild noise
            noise = torch.randn_like(emg_a) * 0.05
            emg_p = emg_a + noise 
            
        # 3. Mine Negative (DIFFERENT text)
        neg_text = random.choice(list(self.sentence_map.keys()))
        while neg_text == text:
            neg_text = random.choice(list(self.sentence_map.keys()))
            
        neg_block, neg_idx = random.choice(self.sentence_map[neg_text])
        emg_n, _ = self._load_tensors(neg_block, neg_idx)
        
        return emg_a, mfcc_a, emg_p, emg_n


def triplet_collate_fn(batch):
    """Pads sequences to the max length within the batch."""
    emg_a_list, mfcc_a_list, emg_p_list, emg_n_list = [], [], [], []
    
    for emg_a, mfcc_a, emg_p, emg_n in batch:
        emg_a_list.append(emg_a)
        mfcc_a_list.append(mfcc_a)
        emg_p_list.append(emg_p)
        emg_n_list.append(emg_n)
        
    emg_a_padded = pad_sequence(emg_a_list, batch_first=True)
    mfcc_a_padded = pad_sequence(mfcc_a_list, batch_first=True)
    emg_p_padded = pad_sequence(emg_p_list, batch_first=True)
    emg_n_padded = pad_sequence(emg_n_list, batch_first=True)
    
    return emg_a_padded, mfcc_a_padded, emg_p_padded, emg_n_padded