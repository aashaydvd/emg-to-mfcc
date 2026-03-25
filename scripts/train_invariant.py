import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# Internal project imports
from src.models.contrastive_model import EMGContrastiveModel
from src.data.dataset_triplets import EMGContrastiveDataset, triplet_collate_fn
from src.data.dataset_training import StandardEMGDataset, baseline_collate_fn
from src.losses.contrastive import TripletContrastiveLoss


def train_invariant_model(lambda_weight=0.5, batch_size=16, num_epochs=50, lr=1e-3, speaker="Spk1"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Hybrid Training Pipeline on {device}")
    print(f"⚙️ Hyperparameters: Batch Size={batch_size}, Epochs={num_epochs}, Lambda={lambda_weight}")

    # 1. DATALOADERS
    # Training: Yields (Anchor EMG, Anchor MFCC, Positive EMG, Negative EMG)
    train_dataset = EMGContrastiveDataset(processed_dir="CSL-EMG_Processed", speaker=speaker)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=triplet_collate_fn)

    # 2. MODEL & LOSSES
    model = EMGContrastiveModel().to(device)
    
    # MSE for acoustic reconstruction (captures all 13 coefficients, including volume)
    recon_criterion = nn.MSELoss()
    
    # Triplet Loss with mandatory L2 Normalization on the projections
    contrastive_criterion = TripletContrastiveLoss(margin=1.0, normalize_embeddings=True) 
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. TRAINING LOOP
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_recon, total_triplet = 0.0, 0.0, 0.0
        
        for emg_a, mfcc_a, emg_p, emg_n in train_loader:
            # Move to device
            emg_a, mfcc_a = emg_a.to(device), mfcc_a.to(device)
            emg_p, emg_n = emg_p.to(device), emg_n.to(device)
            
            optimizer.zero_grad()
            
            # --- FORWARD PASS ---
            # Anchor: Extract acoustic prediction and projected physiological latent
            mfcc_pred_a, z_proj_anchor = model(emg_a, return_latent=True)
            
            # Positive & Negative: Only extract projected physiological latent
            _, z_proj_pos = model(emg_p, return_latent=True)
            _, z_proj_neg = model(emg_n, return_latent=True)
            
            # --- HYBRID LOSS COMPUTATION ---
            # 1. Reconstruction Loss (MSE)
            loss_recon = recon_criterion(mfcc_pred_a, mfcc_a)
            
            # 2. Contrastive Loss (Triplet Margin)
            loss_triplet = contrastive_criterion(z_proj_anchor, z_proj_pos, z_proj_neg)
            
            # 3. Combined Objective
            loss = loss_recon + (lambda_weight * loss_triplet)
            
            # --- BACKPROPAGATION ---
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_triplet += loss_triplet.item()
            

        # --- EPOCH LOGGING ---
        batches = len(train_loader)
        
        print(f"Epoch [{epoch+1:02d}/{num_epochs}] | "
              f"Loss: {total_loss/batches:.3f} | "
              f"MSE: {total_recon/batches:.3f} | "
              f"Triplet: {total_triplet/batches:.4f}")

    torch.save(model.state_dict(), f"models/{speaker}_invariant_model.pth")
    print("Training complete. Weights securely saved to 'invariant_model.pth'")

if __name__ == "__main__":
    for spk in ["Spk1","Spk2","Spk3","Spk4", "Spk5", "Spk6", "Spk7", "Spk8"]:
        train_invariant_model(lambda_weight=0.5,num_epochs=1, speaker = spk)