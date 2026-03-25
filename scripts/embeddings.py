import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

# Ensure these match your actual file structure
from src.models.baseline_model import EMGBaselineModel
from src.models.contrastive_model import EMGContrastiveModel
from src.data.dataset_training import StandardEMGDataset, baseline_collate_fn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


def get_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for emg_batch, _, _ in dataloader:
            emg_batch = emg_batch.to(device)
            
            if isinstance(model, EMGContrastiveModel):
                # Return latent 'z' before the projection head
                _, z = model(emg_batch, return_latent=True)
            else:
                z_seq = model.encoder(emg_batch)
                z = z_seq.mean(dim=1) # Global Average Pooling
            
            # 🔥 Fix: Force 2D (Batch, Features) to prevent sklearn errors
            z_flat = z.view(z.size(0), -1).cpu().numpy()
            embeddings.append(z_flat)
            
    if not embeddings:
        return None
    return np.concatenate(embeddings, axis=0)

def calculate_metrics(embeddings, labels):
    """Calculates Silhouette Score, Centroid Drift, and Cosine Similarity."""
    # 1. Silhouette Score (Cluster distinctness)
    sil = silhouette_score(embeddings, labels)
    
    # 2. Centroid Drift (Euclidean distance B1 -> B7)
    # Labels: 0=Block1, 2=Block7
    idx_b1 = np.where(labels == 0)[0]
    idx_b7 = np.where(labels == 2)[0]
    
    drift = 0.0
    cosine_sim = 0.0
    
    if len(idx_b1) > 0 and len(idx_b7) > 0:
        c1 = embeddings[idx_b1].mean(axis=0).reshape(1, -1)
        c7 = embeddings[idx_b7].mean(axis=0).reshape(1, -1)
        
        drift = np.linalg.norm(c1 - c7)
        cosine_sim = cosine_similarity(c1, c7)[0][0]
        
    return sil, drift, cosine_sim

def run_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Latent Analysis on {device}...")

    # Config
    processed_dir = "CSL-EMG_Processed"
    speaker = "Spk1"
    blocks = ["Block1-Initial", "Block3-Eval1", "Block7-Eval3"]
    
    all_embeddings_base = []
    all_embeddings_inv = []
    all_labels = []

    # Load Models
    base_model = EMGBaselineModel().to(device)
    base_model.load_state_dict(torch.load("baseline_model.pth", map_location=device))
    
    inv_model = EMGContrastiveModel().to(device)
    inv_model.load_state_dict(torch.load("invariant_model.pth", map_location=device))

    for idx, block in enumerate(blocks):
        # 🔥 Using split="eval" to ensure Blocks 3 and 7 are found
        ds = StandardEMGDataset(root_dir=processed_dir, speaker=speaker, split="eval", blocks=[block])
        
        print(f"🔍 {block}: Found {len(ds)} samples.")
        if len(ds) == 0: continue

        loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=baseline_collate_fn)
        
        emb_base = get_embeddings(base_model, loader, device)
        emb_inv = get_embeddings(inv_model, loader, device)
        
        if emb_base is not None:
            all_embeddings_base.append(emb_base)
            all_embeddings_inv.append(emb_inv)
            all_labels.extend([idx] * len(emb_base))

    # --- Post-Processing & Validation ---
    if len(set(all_labels)) < 2:
        print("❌ Error: Need at least 2 blocks to compare. Check your folder names.")
        return

    # Concatenate
    base_raw = np.concatenate(all_embeddings_base, axis=0)
    inv_raw = np.concatenate(all_embeddings_inv, axis=0)
    labels = np.array(all_labels)

    # 🔥 Normalize both spaces so metrics are comparable
    scaler = StandardScaler()
    base_norm = scaler.fit_transform(base_raw)
    inv_norm = scaler.fit_transform(inv_raw)

    # --- Run Metrics ---
    sil_b, drift_b, cos_b = calculate_metrics(base_norm, labels)
    sil_i, drift_i, cos_i = calculate_metrics(inv_norm, labels)

    print("\n" + "="*40)
    print(f"{'Metric':<20} | {'Baseline':<10} | {'Invariant':<10}")
    print("-" * 40)
    print(f"{'Silhouette (↓)':<20} | {sil_b:>10.4f} | {sil_i:>10.4f}")
    print(f"{'Centroid Drift (↓)':<20} | {drift_b:>10.4f} | {drift_i:>10.4f}")
    print(f"{'Cosine Sim (↑)':<20} | {cos_b:>10.4f} | {cos_i:>10.4f}")
    print("="*40)

    # --- t-SNE Visualization ---
    print("🎨 Generating t-SNE plots...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)-1))
    vis_base = tsne.fit_transform(base_norm)
    vis_inv = tsne.fit_transform(inv_norm)

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['#ff4b4b', '#4bff4b', '#4b4bff'] # R, G, B
    
    for idx, block in enumerate(blocks):
        mask = labels == idx
        ax[0].scatter(vis_base[mask, 0], vis_base[mask, 1], label=block, alpha=0.7, c=colors[idx], edgecolors='w')
        ax[1].scatter(vis_inv[mask, 0], vis_inv[mask, 1], label=block, alpha=0.7, c=colors[idx], edgecolors='w')

    ax[0].set_title(f"Baseline Latent Space\n(Drift: {drift_b:.2f}, CosSim: {cos_b:.2f})")
    ax[1].set_title(f"Invariant Latent Space\n(Drift: {drift_i:.2f}, CosSim: {cos_i:.2f})")
    ax[0].legend(); ax[1].legend()
    
    plt.tight_layout()
    plt.savefig("visualisations/latent_drift_analysis.png", dpi=300)
    print("✅ Analysis complete! Plot saved as 'latent_drift_analysis.png'")

if __name__ == "__main__":
    run_analysis()