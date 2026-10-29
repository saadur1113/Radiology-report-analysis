"""
Stage 2: FAST RAG Training with Cached Retrievals

This version uses pre-cached retrievals = 60x faster!

Expected time: 18 hours (vs 22.5 days with on-the-fly retrieval)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import argparse
from tqdm import tqdm
import os
import json
import sys


from modified_qformer_opt import RAGEnhancedBLIP2_OPT
from dataset_opt_cached import MIMICCXRDatasetCached, collate_fn


def train_stage2_cached(
    stage1_checkpoint,
    train_csv,
    val_csv,
    train_cache,  # ← NEW: cached retrievals
    val_cache,    # ← NEW: cached retrievals
    output_dir,
    batch_size=8,
    epochs=5,
    learning_rate=5e-5,
    gradient_accumulation_steps=1
):
    """
    Fast Stage 2 training with cached retrievals
    """
    
    print("="*80)
    print("STAGE 2: FAST RAG TRAINING (CACHED RETRIEVALS)")
    print("="*80)
    
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config = {
        'stage': 2,
        'model': 'BLIP2-OPT-2.7B-RAG',
        'stage1_checkpoint': stage1_checkpoint,
        'train_csv': train_csv,
        'val_csv': val_csv,
        'train_cache': train_cache,
        'val_cache': val_cache,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'cached_retrieval': True
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load datasets with CACHED retrievals
    print("\nLoading datasets with CACHED retrievals...")
    train_dataset = MIMICCXRDatasetCached(
        csv_file=train_csv,
        retrieval_cache_file=train_cache,
        tokenizer_name='Salesforce/blip2-opt-2.7b',
        max_length=128
    )
    
    val_dataset = MIMICCXRDatasetCached(
        csv_file=val_csv,
        retrieval_cache_file=val_cache,
        tokenizer_name='Salesforce/blip2-opt-2.7b',
        max_length=128
    )
    
    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")
    
    # DataLoaders
    print("\nCreating DataLoaders...")
    print(f"Batch size: {batch_size}")
    print(f"num_workers: 4 (safe with cached retrievals!)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Can use >0 with cached data!
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
    
    expected_batches = len(train_dataset) // batch_size
    actual_batches = len(train_loader)
    
    print(f"\nDataLoader Info:")
    print(f"  Expected batches: {expected_batches}")
    print(f"  Actual batches: {actual_batches}")
    print(f"  ✓ Looks good!")
    
    # Initialize model
    print("\nInitializing RAG-enhanced model...")
    model = RAGEnhancedBLIP2_OPT(
        blip2_model_name='Salesforce/blip2-opt-2.7b',
        num_query_tokens=32,
        freeze_vision=True,
        freeze_llm=True,
        use_retrieval=True  # Enable retrieval components
    )
    model = model.to(device)
    
   # Load checkpoint and detect resume
    print(f"\nLoading checkpoint: {stage1_checkpoint}")
    checkpoint = torch.load(stage1_checkpoint, map_location=device)
    
    # Check if this is a resume (has epoch info) or fresh start (Stage 1)
    start_epoch = 0
    best_val_loss_resume = float('inf')
    global_step_start = 0
    
    if 'epoch' in checkpoint and 'val_loss' in checkpoint:
        # This is a Stage 2 checkpoint - RESUME!
        start_epoch = checkpoint['epoch']
        best_val_loss_resume = checkpoint.get('val_loss', float('inf'))
        global_step_start = checkpoint.get('global_step', 0)
        print(f"\n🔄 RESUMING from Epoch {start_epoch}")
        print(f"  Previous val loss: {checkpoint['val_loss']:.4f}")
        print(f"  Will train epochs {start_epoch + 1} to {epochs}")
    else:
        print(f"\n🆕 FRESH START from Stage 1 checkpoint")
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"✓ Model weights loaded")
    
    if start_epoch == 0:
        print(f"  Missing keys: {len(missing_keys)} (retrieval components - normal!)")
        
        # FIX 2: Pre-initialize retrieval cross-attention from visual cross-attention
        print("\n  FIX 2: Pre-initializing retrieval from visual weights...")
        sd = model.state_dict()
        copied = 0
        for key in list(sd.keys()):
            if 'retrieval_cross_attention' in key:
                # Find corresponding visual cross-attention key
                visual_key = key.replace('retrieval_cross_attention.cross_attention.', 'cross_attention.').replace('retrieval_cross_attention.layer_norm.', 'cross_attn_layer_norm.')
                if visual_key in sd and sd[visual_key].shape == sd[key].shape:
                    sd[key] = sd[visual_key].clone()
                    copied += 1
        
        if copied > 0:
            model.load_state_dict(sd)
            print(f"  ✓ Copied {copied} weight tensors from visual → retrieval cross-attention")
        else:
            # Try alternative naming
            for key in list(sd.keys()):
                if 'retrieval' in key and 'cross_attention' in key:
                    visual_key = key.replace('retrieval', 'visual').replace('retrieval_cross_attention', 'cross_attention')
                    # Try multiple naming patterns
                    for pattern in [
                        key.replace('retrieval_cross_attention', 'cross_attention'),
                        key.replace('retrieval_', ''),
                    ]:
                        if pattern in sd and sd[pattern].shape == sd[key].shape:
                            sd[key] = sd[pattern].clone()
                            copied += 1
                            break
            if copied > 0:
                model.load_state_dict(sd)
                print(f"  ✓ Copied {copied} weight tensors (alternative naming)")
            else:
                print(f"  ⚠ Could not find matching visual weights. Listing keys...")
                visual_keys = [k for k in sd.keys() if 'cross_attention' in k and 'retrieval' not in k]
                retrieval_keys = [k for k in sd.keys() if 'retrieval' in k and 'cross_attention' in k]
                print(f"    Visual keys: {visual_keys[:5]}")
                print(f"    Retrieval keys: {retrieval_keys[:5]}")
    
    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    
    # Restore optimizer state if resuming
    if 'optimizer_state_dict' in checkpoint and start_epoch > 0:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Optimizer state restored")
    
    # Scheduler  
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Restore scheduler state if resuming
    if 'scheduler_state_dict' in checkpoint and start_epoch > 0:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✓ Scheduler state restored")
    
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"\nTrainable parameters: {trainable_count:,}")
    
    # Training info
    print("\n" + "="*80)
    print("STARTING FAST TRAINING (CACHED RETRIEVAL)")
    print("="*80)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Total steps: {total_steps}")
    print(f"Expected time: ~18 hours (vs 22.5 days without caching!)")
    print("="*80 + "\n")
    
     # Training loop
    best_val_loss = best_val_loss_resume  # Use resumed value
    global_step = global_step_start  # Use resumed value
    
    log_file = open(os.path.join(output_dir, 'training.log'), 'w')
    
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{epochs} (continuing from checkpoint)")
        print('='*80)
        
        # Training
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        # Force correct progress bar total
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}",
            total=len(train_dataset) // batch_size  # ← Fix progress bar!
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move to device
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                retrieved_texts = batch['retrieved_texts']
                
                # Forward pass
                outputs = model(
                    pixel_values=pixel_values,
                    retrieved_texts=retrieved_texts,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                train_loss += loss.item() * gradient_accumulation_steps
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Log every 50 steps
                if global_step % 50 == 0 and global_step > 0:
                    avg_loss = train_loss / (batch_idx + 1)
                    log_msg = f"Epoch {epoch+1}, Step {global_step}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}\n"
                    log_file.write(log_msg)
                    log_file.flush()
                    print(f"\nStep {global_step}, Loss: {avg_loss:.4f}")
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
        
        # Epoch summary
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
                    retrieved_texts = batch['retrieved_texts']
                    
                    outputs = model(
                        pixel_values=pixel_values,
                        retrieved_texts=retrieved_texts,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
                    
                except Exception as e:
                    print(f"\nValidation error: {e}")
                    continue
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        
        # Save best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            print(f"✓ New best! Saving model...")
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'global_step': global_step,
                'config': config
            }, os.path.join(output_dir, 'best_model.pt'))
        
        # Epoch checkpoint DISABLED to save disk space
        
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
    print("✓ STAGE 2 TRAINING COMPLETE!")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}/best_model.pt")


def main():
    parser = argparse.ArgumentParser(description='Fast Stage 2 with Cached Retrieval')
    
    # Paths
    parser.add_argument('--stage1_checkpoint', type=str, required=True)
    parser.add_argument('--train_csv', type=str, 
                       default='./data/processed/mimic_cxr_train.csv')
    parser.add_argument('--val_csv', type=str,
                       default='./data/processed/mimic_cxr_val.csv')
    parser.add_argument('--train_cache', type=str, required=True,
                       help='Path to cached train retrievals (.pkl)')
    parser.add_argument('--val_cache', type=str, required=True,
                       help='Path to cached val retrievals (.pkl)')
    parser.add_argument('--output_dir', type=str,
                       default='./checkpoints/stage2_opt_fast')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.stage1_checkpoint):
        print(f"Error: Stage 1 checkpoint not found: {args.stage1_checkpoint}")
        return
    
    if not os.path.exists(args.train_cache):
        print(f"Error: Train cache not found: {args.train_cache}")
        print("Run cache_retrievals.py first!")
        return
    
    if not os.path.exists(args.val_cache):
        print(f"Error: Val cache not found: {args.val_cache}")
        print("Run cache_retrievals.py first!")
        return
    
    # Run training
    train_stage2_cached(
        stage1_checkpoint=args.stage1_checkpoint,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        train_cache=args.train_cache,
        val_cache=args.val_cache,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )


if __name__ == '__main__':
    main()
