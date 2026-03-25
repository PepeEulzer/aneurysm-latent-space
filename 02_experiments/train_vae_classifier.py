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
from src.pointnet_vae import VAEEncoder, VAEDecoder
from src.utils import weights_init_vae

# --- Setup Device ---
if torch.cuda.is_available():
    device = torch.device("cuda:1") 
    torch.cuda.set_device(device) 
else:
    device = torch.device("cpu")

# --- Config ---
model_type = "multitask_pointnet_vae_mlp"
loss_type = "pwd_kld_cls" 
data_dirs = ["./datasets/aneurysm_objs_2956/"] # Add other dirs if needed
rupture_labels_file_path = "./datasets/rupture_labels.csv"

batch_size = 32
num_workers = 1 
train_split_percentage = 0.8

# VAE & Classifier Params
beta = 0.00001
sigma_scaling = 1.0 
target_cls_weight = 1.0 # Multiplier to balance Classification Loss vs VAE Loss
num_classes = 2

results_dir = "./results/"
checkpoint_path = "./checkpoints/" 
model_encoder_checkpoint_path = "" 
model_decoder_checkpoint_path = "" 

# Ensure directories exist
os.makedirs(results_dir, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)

# Loop Params
out_dims = [2956] 
z_sizes = [2, 4, 8, 16] 
use_bias = True
max_epochs = 50000 
patience = 500 #1000 # Early stopping patience
num_warmup_epochs = 100 #100

# --- Run Training Loops ---
for out_dim in out_dims:
    for z_size in z_sizes:
        print(f"\n{'='*60}\nTraining {model_type} | out_dim: {out_dim} | z_size: {z_size}\n{'='*60}")

        data_dir = next((d for d in data_dirs if str(out_dim) in d), "")
        assert data_dir != "", f"Could not find data_dir for out_dim {out_dim}"

        loss_curve_path = os.path.join(results_dir, f"{model_type}_{z_size}_{out_dim}_{loss_type}_loss_curve.npy")
        loss_per_epoch = np.load(loss_curve_path).tolist() if (model_encoder_checkpoint_path != "" and os.path.exists(loss_curve_path)) else []

        # --- Setup Train & Val DataLoaders ---
        train_dataset = AneurysmDataset(data_dir=data_dir, rupture_labels_file_path=rupture_labels_file_path, split="train", train_split_percentage=train_split_percentage)
        val_dataset = AneurysmDataset(data_dir=data_dir, rupture_labels_file_path=rupture_labels_file_path, split="val", train_split_percentage=train_split_percentage)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)

        # Setup model (Assumes VAEEncoder accepts num_classes to initialize the classifier)
        model_encoder = VAEEncoder(z_size=z_size, use_bias=use_bias).to(device)
        model_decoder = VAEDecoder(z_size=z_size, use_bias=use_bias, out_dim=out_dim).to(device)
        
        model_decoder.apply(weights_init_vae)
        model_encoder.apply(weights_init_vae)

        if model_encoder_checkpoint_path != "" and os.path.exists(model_encoder_checkpoint_path):
            model_encoder.load_state_dict(torch.load(model_encoder_checkpoint_path, map_location=device))
            model_decoder.load_state_dict(torch.load(model_decoder_checkpoint_path, map_location=device))
            print("Loaded models from checkpoints.")
        
        # Setup optimizer and Classifier Loss
        learning_rate = 0.001 
        grad_clip = 1.0
        # Combine both models' parameters for the optimizer
        optimizer = optim.Adam(params=chain(model_encoder.parameters(), model_decoder.parameters()), lr=learning_rate, betas=(0.9, 0.999))
        criterion_cls = nn.CrossEntropyLoss().to(device)

        # --- Early Stopping Variables ---
        min_val_loss = float('inf') 
        epochs_no_improve = 0

        for epoch in range(1, max_epochs + 1):
            start_epoch_time = datetime.now()

            # Warm-up: No classification loss for the first 50 epochs, 
            # then scale it up to the target weight (e.g., 10.0)
            if epoch < num_warmup_epochs:
                cls_weight = 0.0
            else:
                # Optional: smoothly increase it, or just snap it to target
                cls_weight = target_cls_weight 

            # -------------------------
            # 1. TRAINING PHASE
            # -------------------------
            model_encoder.train()
            model_decoder.train()

            total_train_loss, total_pwd_loss, total_kl_loss, total_cls_loss = 0.0, 0.0, 0.0, 0.0
            valid_cls_batches = 0 

            for i, (point_data, labels) in enumerate(train_loader, 1):
                optimizer.zero_grad()
                
                X = point_data.to(device)
                labels = labels.to(device)

                if X.size(-1) == 3: 
                    X = X.transpose(X.dim() - 2, X.dim() - 1)

                # 1. Forward pass through Encoder
                _, mu_gold, logvar_gold = model_encoder(X)
                
                # 2. Reparameterization Trick
                sigmas = torch.exp(logvar_gold * 0.5)
                if sigma_scaling > 0.0:
                    epsilon = torch.randn_like(sigmas) 
                    z_gold = (epsilon * sigmas * sigma_scaling) + mu_gold
                else:
                    z_gold = mu_gold
                
                # 3. Forward pass through Decoder
                X_rec = model_decoder((z_gold, mu_gold, logvar_gold))

                X = X.permute(0,2,1)
                X_rec = X_rec.permute(0,2,1)
                
                # 4. Calculate VAE Losses (on ALL samples in batch)
                pwd_loss = torch.mean(torch.norm(X_rec - X, p=2, dim=2)) 
                
                mu2 = torch.pow(mu_gold, 2) 
                sigmas2 = torch.exp(logvar_gold)
                kld_loss = beta * torch.sum(mu2 + sigmas2 - logvar_gold - 1)

                # 5. Forward pass through Classifier (Using noisy z_gold)
                predicted_class_logits = model_encoder.classifier(z_gold)
                
                # 6. Calculate Masked Classification Loss
                valid_mask = (labels != 2)
                
                if valid_mask.any():
                    valid_labels = labels[valid_mask]
                    valid_logits = predicted_class_logits[valid_mask]
                    cls_loss = criterion_cls(valid_logits, valid_labels)
                    
                    # Total Loss = VAE Losses + Weighted Classification Loss
                    loss = pwd_loss + kld_loss + (cls_weight * cls_loss)
                    
                    total_cls_loss += cls_loss.item()
                    valid_cls_batches += 1
                else:
                    # If batch is entirely ignored labels, train VAE only
                    loss = pwd_loss + kld_loss 

                # Backprop
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(chain(model_encoder.parameters(), model_decoder.parameters()), max_norm=grad_clip)
                optimizer.step()

                total_train_loss += loss.item()
                total_pwd_loss += pwd_loss.item()
                total_kl_loss += kld_loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_cls_loss = total_cls_loss / valid_cls_batches if valid_cls_batches > 0 else 0.0
            loss_per_epoch.append(avg_train_loss)

            # -------------------------
            # 2. VALIDATION PHASE
            # -------------------------
            model_encoder.eval()
            model_decoder.eval()
            total_val_loss, total_val_pwd, total_val_kld, total_val_cls = 0.0, 0.0, 0.0, 0.0
            val_valid_cls_batches = 0

            with torch.no_grad():
                for point_data, labels in val_loader:
                    X = point_data.to(device)
                    labels = labels.to(device)

                    if X.size(-1) == 3: 
                        X = X.transpose(X.dim() - 2, X.dim() - 1)

                    _, mu_val, logvar_val = model_encoder(X)
                    
                    # For validation, use deterministic mean (mu) for both tasks
                    z_val = mu_val
                    X_rec_val = model_decoder((z_val, mu_val, logvar_val))

                    X = X.permute(0,2,1)
                    X_rec_val = X_rec_val.permute(0,2,1)
                    
                    # VAE Val Losses
                    val_pwd_loss = torch.mean(torch.norm(X_rec_val - X, p=2, dim=2))
                    val_kld_loss = beta * torch.sum(torch.pow(mu_val, 2) + torch.exp(logvar_val) - logvar_val - 1)
                    
                    # Classifier Val Loss
                    predicted_class_logits = model_encoder.classifier(z_val)
                    valid_mask = (labels != 2)

                    if valid_mask.any():
                        valid_labels = labels[valid_mask]
                        valid_logits = predicted_class_logits[valid_mask]
                        val_cls_loss = criterion_cls(valid_logits, valid_labels)
                        
                        val_loss = val_pwd_loss + val_kld_loss + (cls_weight * val_cls_loss)
                        
                        total_val_cls += val_cls_loss.item()
                        val_valid_cls_batches += 1
                    else:
                        val_loss = val_pwd_loss + val_kld_loss

                    total_val_loss += val_loss.item()
                    total_val_pwd += val_pwd_loss.item()
                    total_val_kld += val_kld_loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            avg_val_cls_loss = total_val_cls / val_valid_cls_batches if val_valid_cls_batches > 0 else 0.0

            # -------------------------
            # 3. LOGGING & EARLY STOPPING
            # -------------------------
            if epoch % 10 == 0 or epoch == 1:
                log_str = (f'Epoch: {epoch:04d} | '
                           f'Train Total Loss: {avg_train_loss:.4f} (PWD: {total_pwd_loss/len(train_loader):.4f}, KLD: {total_kl_loss/len(train_loader):.4f}, CLS: {avg_cls_loss:.4f}) | '
                           f'Val Total Loss: {avg_val_loss:.4f} (PWD: {total_val_pwd/len(val_loader):.4f}, KLD: {total_val_kld/len(val_loader):.4f}, CLS: {avg_val_cls_loss:.4f}) | '
                           f'Time: {datetime.now().strftime("%H:%M:%S")}')
                print(log_str)
                with open(os.path.join(results_dir, f'{model_type}_{z_size}_{out_dim}_{loss_type}_loss_curve.txt'), 'a+') as f:
                    f.write(log_str + '\n')

            # Monitor Validation Loss for Saving Models
            if epoch > num_warmup_epochs: 
                if avg_val_loss < min_val_loss:
                    min_val_loss = avg_val_loss
                    epochs_no_improve = 0 
                    
                    # Save both encoder, decoder, and optimizer
                    torch.save(model_encoder.state_dict(), os.path.join(checkpoint_path, f'{model_type}_{z_size}_{out_dim}_{loss_type}_model_encoder.pth'))
                    torch.save(model_decoder.state_dict(), os.path.join(checkpoint_path, f'{model_type}_{z_size}_{out_dim}_{loss_type}_model_decoder.pth'))
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, f'{model_type}_{z_size}_{out_dim}_{loss_type}_optimizer.pth'))
                    np.save(os.path.join(results_dir, f'{model_type}_{z_size}_{out_dim}_{loss_type}_loss_curve.npy'), loss_per_epoch)
                    
                    #if epoch % 10 == 0:
                    print(f"  --> *** New Min Val Loss: {min_val_loss:.4f}! Models saved. ***")
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"\n[!] EARLY STOPPING TRIGGERED at Epoch {epoch}.")
                    print(f"Validation loss did not improve for {patience} epochs. Best Val Loss was: {min_val_loss:.4f}")
                    break