"""
Generate remaining Stage 1 drafts (from index 25600 onward) and merge
with the already-good first 25600 drafts.

Usage:
    python generate_remaining_drafts.py \
        --checkpoint ./checkpoints/stage1_opt_clean/best_model.pt \
        --csv_file ./data/processed/mimic_cxr_train_clean.csv \
        --good_drafts ./data/gen_then_retrieve/stage1_drafts_train_first25600.pkl \
        --output_file ./data/gen_then_retrieve/stage1_drafts_train.pkl \
        --start_idx 25600 \
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
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset


from modified_qformer_opt import RAGEnhancedBLIP2_OPT
from dataset_opt import MIMICCXRDataset, collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--good_drafts', type=str, required=True,
                        help='Pickle file with the first 25600 good drafts')
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--start_idx', type=int, default=25600)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # --- Load good drafts ---
    print(f"Loading existing good drafts from {args.good_drafts}...")
    with open(args.good_drafts, 'rb') as f:
        drafts = pickle.load(f)
    print(f"  Loaded {len(drafts)} good drafts (indices 0-{args.start_idx - 1})")

    # --- Load model ---
    print("\nLoading Stage 1 model...")
    model = RAGEnhancedBLIP2_OPT(
        blip2_model_name='Salesforce/blip2-opt-2.7b',
        num_query_tokens=32,
        freeze_vision=True,
        freeze_llm=True,
        use_retrieval=False
    )

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}, "
              f"val_loss: {checkpoint.get('val_loss', '?')}")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded")

    # --- Load dataset (ONLY the remaining subset) ---
    print(f"\nLoading dataset...")
    full_dataset = MIMICCXRDataset(
        csv_file=args.csv_file,
        tokenizer_name='Salesforce/blip2-opt-2.7b',
        max_length=128
    )

    total = len(full_dataset)
    remaining_indices = list(range(args.start_idx, total))
    subset = Subset(full_dataset, remaining_indices)

    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print(f"  Full dataset: {total} samples")
    print(f"  Generating: {len(remaining_indices)} samples (index {args.start_idx} to {total - 1})")
    print(f"  Batches: {len(dataloader)}")

    # --- Generate ---
    print(f"\nGenerating drafts...")

    generation_config = {
        'max_length': 100,
        'num_beams': 5,
        'early_stopping': True,
        'no_repeat_ngram_size': 3,
        'length_penalty': 1.0,
    }

    current_global_idx = args.start_idx
    new_count = 0
    empty_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating remaining drafts")):
            batch_size_actual = len(batch['pixel_values'])

            try:
                pixel_values = batch['pixel_values'].to(device)

                outputs = model.generate(
                    pixel_values=pixel_values,
                    retrieved_texts=None,
                    **generation_config
                )

                predictions = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for i, pred in enumerate(predictions):
                    global_idx = current_global_idx + i
                    drafts[global_idx] = pred
                    new_count += 1
                    if not pred.strip():
                        empty_count += 1

            except Exception as e:
                print(f"\n  Error in batch {batch_idx}: {e}")
                for i in range(batch_size_actual):
                    global_idx = current_global_idx + i
                    drafts[global_idx] = ""
                    new_count += 1
                    empty_count += 1

            current_global_idx += batch_size_actual

            # Save progress every 500 batches
            if (batch_idx + 1) % 500 == 0:
                with open(args.output_file + '.partial', 'wb') as f:
                    pickle.dump(drafts, f)
                print(f"\n  Progress saved: {len(drafts)}/{total} total drafts")

            # GPU cleanup
            if (batch_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()

    # --- Save final ---
    print(f"\nSaving {len(drafts)} total drafts...")
    with open(args.output_file, 'wb') as f:
        pickle.dump(drafts, f)

    # Clean up partial
    partial = args.output_file + '.partial'
    if os.path.exists(partial):
        os.remove(partial)

    # --- Stats ---
    total_empty = sum(1 for v in drafts.values() if not str(v).strip())
    avg_len = sum(len(str(v).split()) for v in drafts.values()) / max(len(drafts), 1)

    print(f"\n{'=' * 70}")
    print(f"DONE")
    print(f"{'=' * 70}")
    print(f"  Total drafts:      {len(drafts)}")
    print(f"  Pre-existing:      {args.start_idx}")
    print(f"  Newly generated:   {new_count}")
    print(f"  New empty/failed:  {empty_count}")
    print(f"  Total empty:       {total_empty} ({100*total_empty/len(drafts):.1f}%)")
    print(f"  Avg word count:    {avg_len:.1f}")
    print(f"  Output:            {args.output_file}")
    print(f"{'=' * 70}")

    # Spot check a few from the new batch
    print(f"\nSpot check (new drafts):")
    for i in [args.start_idx, args.start_idx + 1, args.start_idx + 100, total - 1]:
        if i in drafts:
            print(f"  [{i}]: \"{str(drafts[i])[:80]}\"")


if __name__ == '__main__':
    main()
