"""
Experiment D: Forced-Utilization Stage 2 Training

Two changes from standard Stage 2:
1. Context = sample's own GT findings report (not retrieved reports)
2. Image features masked (blank image through ViT) with probability p

This creates maximum gradient pressure on the retrieval pathway:
- No visual information available (blank image)
- The only route to low loss is reading the GT from retrieval context
- If the pathway trains → starvation confirmed (training problem)
- If it fails → architectural incapacity confirmed

Usage:
    CUDA_VISIBLE_DEVICES=6 python train_stage2_forced_utilization.py \
        --stage1_checkpoint ./checkpoints/stage1_retrained/best_model.pt \
        --train_csv ./data/processed/mimic_cxr_train_clean.csv \
        --val_csv ./data/processed/mimic_cxr_val_clean.csv \
        --output_dir ./checkpoints/expD_forced/stage2 \
        --mask_prob 1.0 \
        --batch_size 8 --epochs 5
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
import torch
import torch.nn as nn
import os
import json
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import random

from modified_qformer_opt import RAGEnhancedBLIP2_OPT


# ============================================================
# Dataset: GT findings as retrieval context + image masking
# ============================================================

class ForcedUtilizationDataset(Dataset):
    """Dataset that provides GT findings as retrieval context
    and optionally masks images."""
    
    def __init__(self, csv_file, tokenizer, max_length=128, mask_prob=1.0):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        # Create a blank image (black) and transform it
        # This goes through the same pipeline as real images
        blank_pil = Image.new('RGB', (224, 224), color=(0, 0, 0))
        self.blank_image = self.transform(blank_pil)
        
        print(f"✓ ForcedUtilizationDataset: {len(self.df)} samples")
        print(f"  Mask probability: {self.mask_prob}")
        print(f"  Context: GT findings (copy task)")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and transform image (or use blank)
        should_mask = random.random() < self.mask_prob
        
        if should_mask:
            image = self.blank_image.clone()
        else:
            try:
                image = Image.open(row['image_path']).convert('RGB')
                image = self.transform(image)
            except Exception:
                image = self.blank_image.clone()
        
        # GT findings as both target AND retrieval context
        findings = str(row.get('findings', ''))
        
        # Tokenize target
        text_with_eos = findings + self.tokenizer.eos_token
        encoded = self.tokenizer(
            text_with_eos,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        # Context = GT findings wrapped in list (same format as cache)
        # Single report instead of k=3, but same interface
        retrieved_texts = [findings]
        
        return {
            'pixel_values': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'retrieved_texts': retrieved_texts,
            'text': findings,
            'masked': should_mask
        }


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'retrieved_texts': [item['retrieved_texts'] for item in batch],
    }


# ============================================================
# Gradient logging utility
# ============================================================

def log_gradient_norms(model, step, log_file):
    """Log gradient norms for retrieval vs visual pathways."""
    retrieval_norms = []
    visual_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if 'retrieval_cross_attention' in name:
                retrieval_norms.append((name.split('.')[-1], grad_norm))
            elif 'cross_attention' in name and 'retrieval' not in name:
                visual_norms.append((name.split('.')[-1], grad_norm))
    
    if retrieval_norms:
        avg_retrieval = np.mean([n for _, n in retrieval_norms])
        avg_visual = np.mean([n for _, n in visual_norms]) if visual_norms else 0
        ratio = avg_retrieval / avg_visual if avg_visual > 0 else float('inf')
        
        log_line = (f"Step {step}: "
                   f"retrieval_grad={avg_retrieval:.6f}, "
                   f"visual_grad={avg_visual:.6f}, "
                   f"ratio={ratio:.4f}")
        print(f"  GRAD: {log_line}")
        log_file.write(log_line + '\n')
        log_file.flush()


# ============================================================
# Weight analysis utility
# ============================================================

def log_weight_stats(model, epoch, log_file, init_weights=None):
    """Log weight statistics for retrieval cross-attention."""
    print(f"\n  === Weight Stats (Epoch {epoch}) ===")
    log_file.write(f"\n=== Weight Stats (Epoch {epoch}) ===\n")
    
    for name, param in model.named_parameters():
        if 'retrieval_cross_attention' in name and 'weight' in name:
            std = param.data.std().item()
            mean_abs = param.data.abs().mean().item()
            
            change_str = ""
            if init_weights and name in init_weights:
                change = (param.data - init_weights[name]).abs().mean().item()
                change_str = f", change_from_init={change:.6f}"
            
            line = f"  {name}: std={std:.6f}, mean_abs={mean_abs:.6f}{change_str}"
            print(line)
            log_file.write(line + '\n')
    
    log_file.flush()


# ============================================================
# Training
# ============================================================

def train_forced_utilization(
    stage1_checkpoint,
    train_csv,
    val_csv,
    output_dir,
    mask_prob=1.0,
    batch_size=8,
    epochs=5,
    learning_rate=5e-5,
    warmup_steps=500,
    gradient_accumulation_steps=1,
    seed=0,
    device_str='cuda:0',
):
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("EXPERIMENT D: FORCED UTILIZATION")
    print("=" * 80)
    print(f"  Mask probability: {mask_prob}")
    print(f"  Context: GT findings (copy task)")
    print(f"  Stage 1 checkpoint: {stage1_checkpoint}")
    print("=" * 80)
    
    # Save config
    config = {
        'experiment': 'D_forced_utilization',
        'stage1_checkpoint': stage1_checkpoint,
        'mask_prob': mask_prob,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'seed': seed,
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load model
    print("\nLoading model...")
    model = RAGEnhancedBLIP2_OPT(
        blip2_model_name='Salesforce/blip2-opt-2.7b',
        num_query_tokens=32,
        freeze_vision=True,
        freeze_llm=True,
        use_retrieval=True
    )
    
    # Load Stage 1 checkpoint
    print(f"Loading Stage 1 weights from {stage1_checkpoint}")
    checkpoint = torch.load(stage1_checkpoint, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    
    # Save initial retrieval weights for comparison
    init_weights = {}
    for name, param in model.named_parameters():
        if 'retrieval_cross_attention' in name:
            init_weights[name] = param.data.clone()
    
    # Datasets
    print("\nLoading datasets...")
    train_dataset = ForcedUtilizationDataset(
        csv_file=train_csv,
        tokenizer=model.tokenizer,
        max_length=128,
        mask_prob=mask_prob
    )
    val_dataset = ForcedUtilizationDataset(
        csv_file=val_csv,
        tokenizer=model.tokenizer,
        max_length=128,
        mask_prob=mask_prob
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate, weight_decay=0.05
    )
    
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )
    
    # Logging
    log_file = open(os.path.join(output_dir, 'training.log'), 'w')
    grad_log = open(os.path.join(output_dir, 'gradient_norms.log'), 'w')
    weight_log = open(os.path.join(output_dir, 'weight_stats.log'), 'w')
    
    # Log initial weights
    log_weight_stats(model, 0, weight_log, init_weights)
    
    best_val_loss = float('inf')
    global_step = 0
    
    print(f"\nTotal steps: {total_steps}")
    print(f"Training...\n")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                retrieved_texts = batch['retrieved_texts']
                
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    retrieved_texts=retrieved_texts,
                )
                
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                
                # Log gradient norms every 50 steps
                if global_step % 50 == 0:
                    log_gradient_norms(model, global_step, grad_log)
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                train_loss += outputs.loss.item()
                global_step += 1
                
                if global_step % 50 == 0:
                    avg_loss = train_loss / (batch_idx + 1)
                    lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix(
                        loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}"
                    )
                    log_file.write(
                        f"Epoch {epoch+1}, Step {global_step}, "
                        f"Loss: {avg_loss:.4f}, LR: {lr:.2e}\n"
                    )
                    log_file.flush()
                    
            except Exception as e:
                print(f"Error at batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
        
        # Log weight stats per epoch
        log_weight_stats(model, epoch + 1, weight_log, init_weights)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    pixel_values = batch['pixel_values'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    retrieved_texts = batch['retrieved_texts']
                    
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        retrieved_texts=retrieved_texts,
                    )
                    val_loss += outputs.loss.item()
                except Exception as e:
                    continue
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        log_file.write(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}\n")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"✓ New best! Saving model...")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config
            }, os.path.join(output_dir, 'best_model.pt'))
    
    # Final weight comparison
    print("\n" + "=" * 80)
    print("FINAL WEIGHT ANALYSIS")
    print("=" * 80)
    for name, param in model.named_parameters():
        if 'retrieval_cross_attention' in name and 'weight' in name:
            init_std = init_weights[name].std().item()
            final_std = param.data.std().item()
            change = (param.data - init_weights[name]).abs().mean().item()
            print(f"  {name}:")
            print(f"    Init std: {init_std:.6f}")
            print(f"    Final std: {final_std:.6f}")
            print(f"    Mean abs change: {change:.6f}")
    
    print(f"\n{'=' * 80}")
    print(f"✓ TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}/best_model.pt")
    print(f"Gradient log: {output_dir}/gradient_norms.log")
    print(f"Weight stats: {output_dir}/weight_stats.log")
    print(f"{'=' * 80}")
    
    log_file.close()
    grad_log.close()
    weight_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment D: Forced Utilization Stage 2'
    )
    parser.add_argument('--stage1_checkpoint', type=str, required=True)
    parser.add_argument('--train_csv', type=str,
        default='./data/processed/mimic_cxr_train_clean.csv')
    parser.add_argument('--val_csv', type=str,
        default='./data/processed/mimic_cxr_val_clean.csv')
    parser.add_argument('--output_dir', type=str,
        default='./checkpoints/expD_forced/stage2')
    parser.add_argument('--mask_prob', type=float, default=1.0,
        help='Probability of masking image (1.0 = all masked)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    train_forced_utilization(
        stage1_checkpoint=args.stage1_checkpoint,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        mask_prob=args.mask_prob,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_str=args.device,
    )
