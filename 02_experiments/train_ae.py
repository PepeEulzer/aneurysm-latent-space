import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import chain
from datetime import datetime

from src.dataset import AneurysmDataset
from src.pointnet_ae import AEEncoder, AEDecoder
from src.utils import weights_init_ae

# --- Setup Device ---
if torch.cuda.is_available():
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# --- Config ---
model_type = "pointnet_ae"
loss_type = "pointwise_distance"  
data_dirs = [
    "./datasets/aneurysm_objs_716/",
    "./datasets/aneurysm_objs_2956/",
    "./datasets/aneurysm_objs_12156/",
]
rupture_labels_file_path = "./datasets/rupture_labels.csv"
batch_size = 32  
num_workers = 1  
train_split_percentage = 0.8
results_dir = "./results/"
checkpoint_path = "./checkpoints/"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)

model_encoder_checkpoint_path = ""  
model_decoder_checkpoint_path = ""  

# Model params
out_dims = [2956]  
z_sizes = [2, 4, 8, 16]
use_bias = True
max_epochs = 30000  

# --- Early Stopping Params ---
patience = 1000  # How many epochs to wait for improvement before stopping

# --- Run Training Loops ---
for out_dim in out_dims:
    for z_size in z_sizes:
        print(f"\n{'='*50}\nTraining {model_type} | out_dim: {out_dim} | z_size: {z_size}\n{'='*50}")

        data_dir = next((d for d in data_dirs if str(out_dim) in d), "")
        assert data_dir != "", f"Could not find data_dir for out_dim {out_dim}"

        loss_curve_path = os.path.join(results_dir, f"{model_type}_{z_size}_{out_dim}_{loss_type}_loss_curve.npy")
        loss_per_epoch = np.load(loss_curve_path).tolist() if (model_encoder_checkpoint_path != "" and os.path.exists(loss_curve_path)) else []

        # --- Setup Train & Val DataLoaders ---
        train_dataset = AneurysmDataset(data_dir=data_dir, rupture_labels_file_path=rupture_labels_file_path, split="train", train_split_percentage=train_split_percentage)
        val_dataset = AneurysmDataset(data_dir=data_dir, rupture_labels_file_path=rupture_labels_file_path, split="val", train_split_percentage=train_split_percentage)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)

        # Setup model
        model_encoder = AEEncoder(z_size=z_size, use_bias=use_bias).to(device)
        model_decoder = AEDecoder(z_size=z_size, use_bias=use_bias, out_dim=out_dim).to(device)
        
        model_encoder.apply(weights_init_ae)
        model_decoder.apply(weights_init_ae)

        if model_encoder_checkpoint_path != "" and os.path.exists(model_encoder_checkpoint_path):
            model_encoder.load_state_dict(torch.load(model_encoder_checkpoint_path, map_location=device))
            model_decoder.load_state_dict(torch.load(model_decoder_checkpoint_path, map_location=device))
            print("Loaded model from checkpoints.")

        # Setup optimizer
        learning_rate = 0.001
        grad_clip = 1.0
        optimizer = optim.Adam(params=chain(model_encoder.parameters(), model_decoder.parameters()), lr=learning_rate, betas=(0.9, 0.999))

        # --- Early Stopping Tracking Variables ---
        min_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(1, max_epochs + 1):
            start_epoch_time = datetime.now()

            # -------------------------
            # 1. TRAINING PHASE
            # -------------------------
            model_encoder.train()
            model_decoder.train()
            total_train_loss = 0.0

            for i, (point_data, _) in enumerate(train_loader, 1):
                optimizer.zero_grad()
                X = point_data.to(device)

                if X.size(-1) == 3: X = X.transpose(X.dim() - 2, X.dim() - 1)

                z_gold = model_encoder(X)  
                X_rec = model_decoder(z_gold)  

                X = X.permute(0, 2, 1)
                X_rec = X_rec.permute(0, 2, 1)
                
                loss = torch.mean(torch.norm(X_rec - X, p=2, dim=2))

                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(chain(model_encoder.parameters(), model_decoder.parameters()), max_norm=grad_clip)
                optimizer.step()
                
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            loss_per_epoch.append(avg_train_loss)

            # -------------------------
            # 2. VALIDATION PHASE
            # -------------------------
            model_encoder.eval()
            model_decoder.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for point_data, _ in val_loader:
                    X = point_data.to(device)
                    if X.size(-1) == 3: X = X.transpose(X.dim() - 2, X.dim() - 1)

                    z_val = model_encoder(X)
                    X_rec_val = model_decoder(z_val)

                    X = X.permute(0, 2, 1)
                    X_rec_val = X_rec_val.permute(0, 2, 1)
                    
                    val_loss = torch.mean(torch.norm(X_rec_val - X, p=2, dim=2))
                    total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)

            # -------------------------
            # 3. LOGGING & EARLY STOPPING
            # -------------------------
            if epoch % 10 == 0 or epoch == 1: # Log every 10 epochs to keep console clean
                log_str = f"Epoch: {epoch:04d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {datetime.now() - start_epoch_time}"
                print(log_str)
                with open(os.path.join(results_dir, f"{model_type}_{z_size}_{out_dim}_{loss_type}_loss_curve.txt"), "a+") as f:
                    f.write(log_str + "\n")

            # Check if this is the best model so far
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                epochs_no_improve = 0  # Reset patience counter
                
                # Save the "Best" model!
                torch.save(model_encoder.state_dict(), os.path.join(checkpoint_path, f"{model_type}_{z_size}_{out_dim}_{loss_type}_model_encoder.pth"))
                torch.save(model_decoder.state_dict(), os.path.join(checkpoint_path, f"{model_type}_{z_size}_{out_dim}_{loss_type}_model_decoder.pth"))
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, f"{model_type}_{z_size}_{out_dim}_{loss_type}_optimizer.pth"))
                np.save(os.path.join(results_dir, f"{model_type}_{z_size}_{out_dim}_{loss_type}_loss_curve.npy"), loss_per_epoch)
                
                if epoch % 10 == 0:
                    print(f"  --> *** New Min Val Loss: {min_val_loss:.4f}! Models saved. ***")
            else:
                epochs_no_improve += 1

            # Trigger Early Stopping
            if epochs_no_improve >= patience:
                print(f"\n[!] EARLY STOPPING TRIGGERED at Epoch {epoch}.")
                print(f"Validation loss did not improve for {patience} epochs. Best Val Loss was: {min_val_loss:.4f}")
                break # Exits the epoch loop and moves to the next z_size