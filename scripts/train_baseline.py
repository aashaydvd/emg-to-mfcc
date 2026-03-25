import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
import os

from src.data.dataset_training import StandardEMGDataset, baseline_collate_fn
from src.models.baseline_model import EMGBaselineModel


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 1. Initialize Datasets and Dataloaders
train_dataset = StandardEMGDataset(root_dir="CSL-EMG_Processed", speaker="Spk1", split="train")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=baseline_collate_fn)

# 2. Initialize Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EMGBaselineModel().to(device)

criterion = nn.MSELoss() # Standard reconstruction loss for MFCCs
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3. Training Loop
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (emg_batch, mfcc_batch, lengths) in enumerate(train_loader):
        emg_batch = emg_batch.to(device)
        mfcc_batch = mfcc_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        mfcc_pred = model(emg_batch)
        
        # Calculate loss (ignoring padded regions is best practice)
        # Create a boolean mask based on the actual sequence lengths
        max_len = emg_batch.size(1)
        mask = torch.arange(max_len)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1).expand_as(mfcc_pred).to(device)
        
        # Only compute MSE on the unpadded data
        loss = criterion(mfcc_pred[mask], mfcc_batch[mask])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train MSE Loss: {avg_loss:.4f}")


torch.save(model.state_dict(), 'models/baseline_model.pth')
