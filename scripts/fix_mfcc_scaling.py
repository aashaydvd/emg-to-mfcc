import os
import numpy as np
from tqdm import tqdm

ROOT_DIR = "CSL-EMG_Processed"

def fix_all_mfccs():
    print(f"🛠️ Starting In-Place MFCC Scaling Fix in {ROOT_DIR}...")
    
    # Walk through every file in the processed directory
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            if file.endswith("_mfcc.npy"):
                file_path = os.path.join(root, file)
                
                # Load
                data = np.load(file_path)
                
                # Apply Z-score Normalization
                # (We use per-sample normalization here for simplicity and speed)
                mean = data.mean()
                std = data.std() + 1e-8
                normalized_data = (data - mean) / std
                
                # Overwrite the original file
                np.save(file_path, normalized_data)

    print("✅ All MFCCs have been normalized to Mean=0, Std=1.")

if __name__ == "__main__":
    fix_all_mfccs()