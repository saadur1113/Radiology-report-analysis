"""
Step 1: Generate Stage 1 Drafts for Training Set

Runs Stage 1 inference on the training set to produce draft reports.
These drafts will be used as retrieval queries in the generate-then-retrieve pipeline.

Usage:
    python generate_stage1_drafts.py \
        --checkpoint ./checkpoints/stage1_opt_clean/best_model.pt \
        --csv_file ./data/processed/mimic_cxr_train_clean.csv \
        --output_file ./data/gen_then_retrieve/stage1_drafts_train.pkl \
        --batch_size 16 \
        --device cuda:0

    # Also generate for val set (even if you have eval outputs, 
    # we need them in the same format):
    python generate_stage1_drafts.py \
        --checkpoint ./checkpoints/stage1_opt_clean/best_model.pt \
        --csv_file ./data/processed/mimic_cxr_val_clean.csv \
        --output_file ./data/gen_then_retrieve/stage1_drafts_val.pkl \
        --batch_size 16 \
        --device cuda:0
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
import os
import sys
import torch
import argparse
import pickle
import json
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader


from modified_qformer_opt import RAGEnhancedBLIP2_OPT
from dataset_opt import MIMICCXRDataset, collate_fn


def generate_drafts(checkpoint_path, csv_file, output_file, batch_size=16, device='cuda:0'):
    """
    Generate Stage 1 draft reports for all samples in csv_file.
    
    Saves a dict: { dataset_index: generated_text }
    """
    
    print("=" * 80)
    print("GENERATING STAGE 1 DRAFTS")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"CSV file:   {csv_file}")
    print(f"Output:     {output_file}")
    print(f"Batch size: {batch_size}")
    print(f"Device:     {device}")
    print("=" * 80)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # --- Load Model ---
    print("\n[1/4] Loading Stage 1 model...")
    model = RAGEnhancedBLIP2_OPT(
        blip2_model_name='Salesforce/blip2-opt-2.7b',
        num_query_tokens=32,
        freeze_vision=True,
        freeze_llm=True,
        use_retrieval=False  # Stage 1 = no retrieval
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}, "
              f"val_loss: {checkpoint.get('val_loss', '?')}")
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # --- Load Dataset ---
    print("\n[2/4] Loading dataset...")
    dataset = MIMICCXRDataset(
        csv_file=csv_file,
        tokenizer_name='Salesforce/blip2-opt-2.7b',
        max_length=128
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # IMPORTANT: keep order for index mapping
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"✓ Dataset: {len(dataset)} samples, {len(dataloader)} batches")
    
    # --- Check for resumability ---
    drafts = {}
    progress_file = output_file + '.progress'
    start_idx = 0
    
    if os.path.exists(progress_file):
        print(f"\n  Found progress file, loading...")
        with open(progress_file, 'rb') as f:
            drafts = pickle.load(f)
        start_idx = len(drafts)
        print(f"  Resuming from sample {start_idx}/{len(dataset)}")
    
    # --- Generate ---
    print(f"\n[3/4] Generating drafts...")
    
    generation_config = {
        'max_length': 100,
        'num_beams': 5,
        'early_stopping': True,
        'no_repeat_ngram_size': 3,
        'length_penalty': 1.0,
    }
    
    current_idx = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating drafts")):
            batch_size_actual = len(batch['pixel_values'])
            
            # Skip already processed batches
            if current_idx + batch_size_actual <= start_idx:
                current_idx += batch_size_actual
                continue
            
            try:
                pixel_values = batch['pixel_values'].to(device)
                
                outputs = model.generate(
                    pixel_values=pixel_values,
                    retrieved_texts=None,
                    **generation_config
                )
                
                predictions = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for i, pred in enumerate(predictions):
                    idx = current_idx + i
                    if idx not in drafts:  # Don't overwrite if resuming
                        drafts[idx] = pred
                
            except Exception as e:
                print(f"\n  Error in batch {batch_idx}: {e}")
                for i in range(batch_size_actual):
                    idx = current_idx + i
                    if idx not in drafts:
                        drafts[idx] = ""  # Empty string for failed samples
            
            current_idx += batch_size_actual
            
            # Save progress every 200 batches
            if (batch_idx + 1) % 200 == 0:
                with open(progress_file, 'wb') as f:
                    pickle.dump(drafts, f)
                print(f"\n  Progress saved: {len(drafts)}/{len(dataset)} samples")
            
            # GPU cleanup
            if (batch_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()
    
    # --- Save final output ---
    print(f"\n[4/4] Saving {len(drafts)} drafts...")
    
    with open(output_file, 'wb') as f:
        pickle.dump(drafts, f)
    print(f"✓ Saved to {output_file}")
    
    # Clean up progress file
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    # --- Quick stats ---
    non_empty = sum(1 for v in drafts.values() if v.strip())
    avg_len = sum(len(v.split()) for v in drafts.values()) / max(len(drafts), 1)
    
    print(f"\n{'=' * 80}")
    print(f"✅ DRAFT GENERATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"  Total drafts:    {len(drafts)}")
    print(f"  Non-empty:       {non_empty}")
    print(f"  Empty/failed:    {len(drafts) - non_empty}")
    print(f"  Avg word count:  {avg_len:.1f}")
    print(f"  Output:          {output_file}")
    print(f"{'=' * 80}")
    
    # Print a few examples
    print("\nSample drafts:")
    for i in range(min(3, len(drafts))):
        text = drafts[i][:150]
        print(f"  [{i}]: {text}...")
    
    return drafts


def main():
    parser = argparse.ArgumentParser(description='Generate Stage 1 drafts')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Stage 1 checkpoint')
    parser.add_argument('--csv_file', type=str, required=True,
                        help='CSV file (train or val)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output pickle file for drafts')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return
    if not os.path.exists(args.csv_file):
        print(f"ERROR: CSV not found: {args.csv_file}")
        return
    
    generate_drafts(
        checkpoint_path=args.checkpoint,
        csv_file=args.csv_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == '__main__':
    main()
