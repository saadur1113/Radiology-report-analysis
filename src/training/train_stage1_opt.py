"""
Stage 1: Medical Domain Adaptation for OPT-based BLIP2

Trains the Q-Former to adapt to medical imaging
No retrieval in this stage - pure vision-language alignment

Features:
- Automatic checkpointing every epoch
- Resumable from checkpoint
- GPU memory optimization
- Progress tracking
- Error recovery
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import argparse
from datetime import datetime
import json

# Import OPT version
from modified_qformer_opt import RAGEnhancedBLIP2_OPT
from dataset_opt import MIMICCXRDataset, collate_fn


def save_checkpoint(
    model, optimizer, scheduler, epoch, step, loss, checkpoint_dir, is_best=False
):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save latest
    latest_path = os.path.join(checkpoint_dir, 'latest.pt')
    torch.save(checkpoint, latest_path)
    
    # Skip per-epoch checkpoint to save disk space
    # epoch_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pt')
    # torch.save(checkpoint, epoch_path)
    
    # Save best
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model (loss: {loss:.4f})")
    
    print(f"✓ Checkpoint saved: {latest_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load checkpoint and resume training"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    print(f"✓ Resumed from epoch {epoch}, step {step}, loss {loss:.4f}")
    
    return epoch, step, loss


def train_stage1(
    train_csv,
    val_csv,
    output_dir='./checkpoints/stage1_opt',
    batch_size=8,
    epochs=5,
    learning_rate=1e-4,
    warmup_steps=500,
    gradient_accumulation_steps=1,
    resume_from=None,
    log_interval=50,
    save_interval=1000,
    seed=0,
    device_str='cuda:0',
):
    """
    Stage 1: Medical domain adaptation
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        output_dir: Where to save checkpoints
        batch_size: Batch size per GPU
        epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Warmup steps for scheduler
        gradient_accumulation_steps: Gradient accumulation
        resume_from: Path to checkpoint to resume from
        log_interval: Log every N steps
        save_interval: Save checkpoint every N steps
    """
    
    print("="*80)
    print("STAGE 1: MEDICAL DOMAIN ADAPTATION (OPT)")
    print("="*80)

    # Set random seed for reproducibility
    if seed > 0:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {seed}")

    # Setup
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config = {
        'stage': 1,
        'model': 'BLIP2-OPT-2.7B',
        'train_csv': train_csv,
        'val_csv': val_csv,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'warmup_steps': warmup_steps,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'seed': seed,
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = MIMICCXRDataset(
        csv_file=train_csv,
        tokenizer_name='Salesforce/blip2-opt-2.7b',
        max_length=128
    )
    
    val_dataset = MIMICCXRDataset(
        csv_file=val_csv,
        tokenizer_name='Salesforce/blip2-opt-2.7b',
        max_length=128
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"✓ Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"✓ Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # Initialize model
    print("\nInitializing model...")
    model = RAGEnhancedBLIP2_OPT(
        blip2_model_name='Salesforce/blip2-opt-2.7b',
        num_query_tokens=32,
        freeze_vision=True,
        freeze_llm=True,
        use_retrieval=False  # No retrieval in Stage 1
    )
    
    model = model.to(device)
    
    # Optimizer - only train Q-Former
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    print(f"\nTrainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    
    # Scheduler
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    # Resume if needed
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if resume_from and os.path.exists(resume_from):
        start_epoch, global_step, best_val_loss = load_checkpoint(
            resume_from, model, optimizer, scheduler
        )
        start_epoch += 1  # Start from next epoch
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Total steps: {total_steps}")
    print("="*80)
    
    # Training log
    log_file = open(os.path.join(output_dir, 'training.log'), 'a')
    
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{epochs}")
        print(f"{'='*80}")
        
        # Training
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move to device
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    retrieved_texts=None  # No retrieval in Stage 1
                )
                
                loss = outputs.loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                
                # Track loss
                train_loss += loss.item() * gradient_accumulation_steps
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item() * gradient_accumulation_steps,
                    'lr': optimizer.param_groups[0]['lr']
                })
                
                # Logging
                if global_step % log_interval == 0:
                    avg_loss = train_loss / (batch_idx + 1)
                    log_msg = f"Epoch {epoch+1}, Step {global_step}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}\n"
                    log_file.write(log_msg)
                    log_file.flush()
                
                # Save checkpoint
                if global_step % save_interval == 0 and global_step > 0:
                    save_checkpoint(
                        model, optimizer, scheduler,
                        epoch, global_step, train_loss / (batch_idx + 1),
                        output_dir
                    )
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                print("Skipping batch and continuing...")
                continue
        
        # Epoch complete
        avg_train_loss = train_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        print("\nRunning validation...")
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    pixel_values = batch['pixel_values'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        retrieved_texts=None
                    )
                    
                    val_loss += outputs.loss.item()
                    
                except Exception as e:
                    print(f"\nValidation error: {e}")
                    continue
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        
        # Save epoch checkpoint
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        
        save_checkpoint(
            model, optimizer, scheduler,
            epoch, global_step, avg_val_loss,
            output_dir, is_best=is_best
        )
        
        # Log epoch summary
        log_msg = f"\nEPOCH {epoch+1} SUMMARY:\n"
        log_msg += f"  Train Loss: {avg_train_loss:.4f}\n"
        log_msg += f"  Val Loss: {avg_val_loss:.4f}\n"
        log_msg += f"  Best Val Loss: {best_val_loss:.4f}\n"
        log_msg += f"  Steps: {global_step}\n\n"
        log_file.write(log_msg)
        log_file.flush()
    
    log_file.close()
    
    print("\n" + "="*80)
    print("✓ STAGE 1 TRAINING COMPLETE!")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}/best_model.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', default='./data/processed/mimic_cxr_train.csv')
    parser.add_argument('--val_csv', default='./data/processed/mimic_cxr_val.csv')
    parser.add_argument('--output_dir', default='./checkpoints/stage1_opt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    train_stage1(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        resume_from=args.resume_from,
        seed=args.seed,
        device_str=args.device
    )


if __name__ == '__main__':
    main()