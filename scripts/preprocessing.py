import os
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path

def preprocess_utterance(emg_path, audio_path):
    """Applies normalizations and MFCC extraction, returning aligned numpy arrays."""
    emg_np = np.load(emg_path)
    audio_np = np.load(audio_path)

    # ---------------------------------------------------------
    # 1. AUDIO PREPROCESSING & MFCC EXTRACTION
    # ---------------------------------------------------------
    audio_signal = audio_np[:, 0]
    
    # Peak Normalization to [-0.99, 0.99]
    max_amp = np.max(np.abs(audio_signal))
    if max_amp > 0:
        audio_signal = (audio_signal / max_amp) * 0.99

    waveform = torch.tensor(audio_signal, dtype=torch.float32).unsqueeze(0)

    # Extract MFCC
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000, n_mfcc=13, melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 40}
    )
    mfcc = mfcc_transform(waveform).squeeze(0).transpose(0, 1).numpy()

    # ---------------------------------------------------------
    # 2. EMG PREPROCESSING & ALIGNMENT
    # ---------------------------------------------------------
    # Z-Score Normalization (Per Channel)
    emg_mean = np.mean(emg_np, axis=0)
    emg_std = np.std(emg_np, axis=0)
    emg_normalized = (emg_np - emg_mean) / (emg_std + 1e-8)

    # Downsample to match MFCC length
    T_mfcc = len(mfcc)
    T_emg, C = emg_normalized.shape
    indices = np.linspace(0, T_emg - 1, T_mfcc)
    
    emg_aligned = np.stack([
        np.interp(indices, np.arange(T_emg), emg_normalized[:, c]) 
        for c in range(C)
    ], axis=1)

    return emg_aligned, mfcc

def process_dataset(raw_dir, out_dir, speaker="Spk1", blocks=None):
    """Iterates through the dataset, processes files, and saves them to a new directory."""
    
    if blocks is None:
        blocks = [
            "Block1-Initial", "Block2-Adapt1", "Block3-Eval1", 
            "Block4-Adapt2", "Block5-Eval2", "Block6-Adapt3", "Block7-Eval3"
        ]

    # Create the base output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    total_processed = 0

    for block in blocks:
        block_folder = f"{speaker}_{block}"
        json_path = os.path.join(raw_dir, block_folder, "recordingLog.json")
        
        if not os.path.exists(json_path):
            print(f"⚠️  Skipping {block_folder}: JSON not found.")
            continue
            
        with open(json_path, 'r') as f:
            log_data = json.load(f)
            
        if "promptList" not in log_data:
            print(f"⚠️  Skipping {block_folder}: 'promptList' missing from JSON.")
            continue
            
        print(f"⚙️  Processing {block_folder}...")
        
        # Also copy the JSON to the new folder so dataloaders can still read the splits!
        out_block_dir = os.path.join(out_dir, block_folder)
        Path(out_block_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(out_block_dir, "recordingLog.json"), 'w') as f:
            json.dump(log_data, f, indent=4)
            
        # Process each utterance
        for idx in log_data["promptList"].keys():
            raw_base = os.path.join(raw_dir, block_folder, str(idx), f"{block_folder}_{idx}")
            emg_raw_path = f"{raw_base}_emg.npy"
            audio_raw_path = f"{raw_base}_audio.npy"
            
            if not os.path.exists(emg_raw_path) or not os.path.exists(audio_raw_path):
                continue
                
            try:
                # 1. Process
                emg_proc, mfcc_proc = preprocess_utterance(emg_raw_path, audio_raw_path)
                
                # 2. Save to new directory
                out_utt_dir = os.path.join(out_block_dir, str(idx))
                Path(out_utt_dir).mkdir(parents=True, exist_ok=True)
                
                out_base = os.path.join(out_utt_dir, f"{block_folder}_{idx}")
                np.save(f"{out_base}_emg_proc.npy", emg_proc)
                np.save(f"{out_base}_mfcc.npy", mfcc_proc)
                
                total_processed += 1
                
            except Exception as e:
                print(f"❌ Error processing {block_folder} ID {idx}: {e}")

    print(f"✅ Preprocessing complete! {total_processed} utterances saved to {out_dir}/")

# --- Run the Script ---
if __name__ == "__main__":
    RAW_DATA_PATH = "CSL-EMG_Array"
    PROCESSED_DATA_PATH = "CSL-EMG_Processed"
    
    process_dataset(raw_dir=RAW_DATA_PATH, out_dir=PROCESSED_DATA_PATH, speaker="Spk1")