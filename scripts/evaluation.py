import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import your models and dataset
from src.models.baseline_model import EMGBaselineModel
from src.models.contrastive_model import EMGContrastiveModel
from src.data.dataset_training import StandardEMGDataset, baseline_collate_fn

def calculate_mcd_tensor(pred, target, lengths):
    """
    Calculates the Mel-Cepstral Distortion (MCD) in dB.
    Standardized for MFCCs (excluding the 0-th energy coefficient).
    """
    # 1. Slice to remove the 0-th coefficient (Index 0 is energy/volume)
    # Shape: (Batch, Time, 12)
    p = pred[:, :, 1:]
    t = target[:, :, 1:]
    
    # 2. Calculate the squared difference: (c_d - \hat{c}_d)^2
    diff_sq = (p - t) ** 2
    
    # 3. Sum across dimensions: \sum_{d=1}^{D-1} (c_d - \hat{c}_d)^2
    # Resulting shape: (Batch, Time)
    sum_diff_sq = torch.sum(diff_sq, dim=-1)
    
    # 4. Standard MCD Formula: (10 / ln 10) * sqrt( 2 * sum_diff_sq )
    # This constant (approx 6.14) converts the distance to decibels.
    k = (10.0 / np.log(10.0)) * np.sqrt(2.0)
    mcd_frames = k * torch.sqrt(sum_diff_sq)
    
    # 5. Masking out padded frames
    max_len = pred.size(1)
    mask = torch.arange(max_len, device=pred.device)[None, :] < lengths[:, None]
    
    if mask.sum() == 0:
        return 0.0
        
    # Average only the valid frames
    return mcd_frames[mask].mean().item()

def evaluate_models(baseline_path="baseline_model.pth", invariant_path="invariant_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 Running Evaluation on {device}")
    
    # 1. Initialize and Load Models
    baseline_model = EMGBaselineModel().to(device)
    baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
    baseline_model.eval()
    
    invariant_model = EMGContrastiveModel().to(device)
    invariant_model.load_state_dict(torch.load(invariant_path, map_location=device))
    invariant_model.eval()

    # The evaluation blocks defined in the CSL-EMG corpus
    eval_blocks = ["Block3-Eval1", "Block5-Eval2", "Block7-Eval3"]
    
    baseline_scores = []
    invariant_scores = []
    
    print("="*65)
    print(f"{'Block':<15} | {'Baseline MCD':<15} | {'Invariant MCD':<15} | {'Improvement':<15}")
    print("="*65)

    for block in eval_blocks:
        dataset = StandardEMGDataset(
            root_dir="CSL-EMG_Processed", 
            speaker="Spk1", 
            split="eval", 
            blocks=[block]
        )
        
        if len(dataset) == 0:
            print(f"⚠️ Skipping {block}: No data found.")
            continue
            
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=baseline_collate_fn)
        
        block_baseline_mcd = 0.0
        block_invariant_mcd = 0.0
        
        with torch.no_grad():
            for emg_batch, mfcc_batch, lengths in dataloader:
                emg_batch = emg_batch.to(device)
                mfcc_batch = mfcc_batch.to(device)
                lengths = lengths.to(device)
                
                # Baseline Inference
                base_pred = baseline_model(emg_batch)
                block_baseline_mcd += calculate_mcd_tensor(base_pred, mfcc_batch, lengths)
                
                # Invariant Inference (return_latent defaults to False)
                inv_pred = invariant_model(emg_batch)
                block_invariant_mcd += calculate_mcd_tensor(inv_pred, mfcc_batch, lengths)
                
        # Average across the batches
        avg_base = block_baseline_mcd / len(dataloader)
        avg_inv = block_invariant_mcd / len(dataloader)
        
        baseline_scores.append(avg_base)
        invariant_scores.append(avg_inv)
        
        # Calculate how much the invariant model improved the score
        diff = avg_base - avg_inv
        improvement = f"-{diff:.3f}" if diff > 0 else f"+{abs(diff):.3f}"
        
        print(f"{block:<15} | {avg_base:<15.3f} | {avg_inv:<15.3f} | {improvement:<15}")

    print("="*65)
    
    # 2. Generate the Comparison Chart
    plot_results(eval_blocks, baseline_scores, invariant_scores)


def plot_results(blocks, baseline_scores, invariant_scores):
    """Generates a line chart to visualize the domain shift and contrastive mitigation."""
    # Simplify labels for the X-axis (e.g., "Block1", "Block3")
    x_labels = [b.split('-')[0] for b in blocks]
    x = np.arange(len(x_labels))
    
    plt.figure(figsize=(10, 6))
    
    # Plot both lines
    plt.plot(x, baseline_scores, marker='o', linestyle='--', color='red', linewidth=2, label='Baseline Model')
    plt.plot(x, invariant_scores, marker='s', linestyle='-', color='blue', linewidth=2, label='Session-Invariant Model')
    
    plt.title('EMG-to-Speech: MCD Degradation Across Time-Drift Blocks', fontsize=14, fontweight='bold')
    plt.xlabel('Recording Session (Temporal Drift)', fontsize=12)
    plt.ylabel('Mel-Cepstral Distortion (MCD) ↓ Lower is better', fontsize=12)
    
    plt.xticks(x, x_labels)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Highlight the gap
    for i in range(len(x)):
        plt.vlines(x=x[i], ymin=invariant_scores[i], ymax=baseline_scores[i], color='gray', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('visualisations/mcd_comparison.png', dpi=300)
    print("📈 Saved comparison chart to 'mcd_comparison.png'")


if __name__ == "__main__":
    # Ensure both models have been trained and weights exist in the current directory
    evaluate_models("baseline_model.pth", "invariant_model.pth")