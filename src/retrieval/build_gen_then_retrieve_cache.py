"""
Step 3: Build Generate-Then-Retrieve Cache

Uses Stage 1 drafts as retrieval queries against a FAISS index of 
GT training reports. Produces retrieval caches compatible with Stage 2 training.

Usage:
    python build_gen_then_retrieve_cache.py \
        --train_csv ./data/processed/mimic_cxr_train_clean.csv \
        --val_csv ./data/processed/mimic_cxr_val_clean.csv \
        --train_drafts ./data/gen_then_retrieve/stage1_drafts_train.pkl \
        --val_drafts ./data/gen_then_retrieve/stage1_drafts_val.pkl \
        --output_dir ./data/gen_then_retrieve \
        --k 3 \
        --encoder_model "pritamdeka/S-PubMedBert-MS-MARCO" \
        --batch_size 64 \
        --device cuda:0
"""

import os
import sys
import torch
import argparse
import pickle
import numpy as np
import faiss
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pandas as pd


def encode_texts(texts, encoder, batch_size=64, desc="Encoding"):
    """Encode a list of texts using sentence transformer"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i + batch_size]
        embeddings = encoder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings).astype('float32')


def build_faiss_index(embeddings):
    """Build a FAISS index from embeddings"""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim with normalized vectors)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    return index


def retrieve_for_drafts(draft_embeddings, index, gt_texts, k=3, self_indices=None):
    """
    Retrieve top-k GT reports for each draft.
    
    Args:
        draft_embeddings: embeddings of Stage 1 drafts
        index: FAISS index of GT training reports
        gt_texts: list of GT training report texts
        k: number of retrievals
        self_indices: if provided, exclude these indices (for train set self-retrieval)
    
    Returns:
        cache dict: { sample_idx: [retrieved_text_1, retrieved_text_2, ...] }
    """
    cache = {}
    
    # Normalize query embeddings
    faiss.normalize_L2(draft_embeddings)
    
    # Retrieve extra if we need to filter self-matches
    k_search = k + 1 if self_indices is not None else k
    
    distances, indices = index.search(draft_embeddings, k_search)
    
    for i in range(len(draft_embeddings)):
        retrieved = []
        for j in range(k_search):
            idx = indices[i][j]
            
            # Skip self-retrieval for training set
            if self_indices is not None and idx == self_indices[i]:
                continue
            
            if idx < len(gt_texts):
                retrieved.append(gt_texts[idx])
            
            if len(retrieved) >= k:
                break
        
        cache[i] = retrieved
    
    return cache


def analyze_retrieval_quality(cache, gt_texts_indexed, drafts_dict):
    """Print retrieval quality stats"""
    
    n_samples = len(cache)
    n_empty = sum(1 for v in cache.values() if not v)
    avg_retrieved = np.mean([len(v) for v in cache.values()])
    
    # Check average similarity of retrieved vs draft
    sample_indices = list(cache.keys())[:100]
    
    print(f"\n  Retrieval Quality:")
    print(f"    Total samples: {n_samples}")
    print(f"    Empty retrievals: {n_empty}")
    print(f"    Avg retrieved per sample: {avg_retrieved:.1f}")
    
    # Show a few examples
    print(f"\n  Sample retrievals:")
    for i in range(min(3, len(sample_indices))):
        idx = sample_indices[i]
        draft = drafts_dict.get(idx, "N/A")[:100]
        retrieved = cache[idx][0][:100] if cache[idx] else "N/A"
        print(f"    [{idx}] Draft:     {draft}...")
        print(f"         Retrieved: {retrieved}...")
        print()


def main():
    parser = argparse.ArgumentParser(description='Build generate-then-retrieve cache')
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, default=None)
    parser.add_argument('--train_drafts', type=str, required=True)
    parser.add_argument('--val_drafts', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./data/gen_then_retrieve')
    parser.add_argument('--k', type=int, default=3, help='Number of retrievals per sample')
    parser.add_argument('--encoder_model', type=str, 
                        default='pritamdeka/S-PubMedBert-MS-MARCO',
                        help='Sentence transformer model for encoding')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("BUILD GENERATE-THEN-RETRIEVE CACHE")
    print("=" * 80)
    print(f"  Encoder:  {args.encoder_model}")
    print(f"  k:        {args.k}")
    print(f"  Device:   {args.device}")
    print("=" * 80)
    
    # --- Load data ---
    print("\n[1/6] Loading data...")
    
    train_df = pd.read_csv(args.train_csv)
    gt_col = 'findings' if 'findings' in train_df.columns else 'report'
    gt_findings_train = [str(x) for x in train_df[gt_col].tolist()]
    print(f"  GT training reports: {len(gt_findings_train)}")
    
    with open(args.train_drafts, 'rb') as f:
        train_drafts = pickle.load(f)
    print(f"  Train drafts: {len(train_drafts)}")
    
    val_drafts = None
    if args.val_drafts and os.path.exists(args.val_drafts):
        with open(args.val_drafts, 'rb') as f:
            val_drafts = pickle.load(f)
        print(f"  Val drafts: {len(val_drafts)}")
    
    # --- Load encoder ---
    print(f"\n[2/6] Loading encoder: {args.encoder_model}")
    encoder = SentenceTransformer(args.encoder_model, device=args.device)
    print(f"  ✓ Encoder loaded (dim={encoder.get_sentence_embedding_dimension()})")
    
    # --- Encode GT training reports ---
    print(f"\n[3/6] Encoding GT training reports...")
    gt_embeddings = encode_texts(
        gt_findings_train, encoder, 
        batch_size=args.batch_size, desc="Encoding GT reports"
    )
    print(f"  ✓ GT embeddings shape: {gt_embeddings.shape}")
    
    # Build FAISS index
    print(f"\n  Building FAISS index...")
    index = build_faiss_index(gt_embeddings.copy())
    print(f"  ✓ FAISS index: {index.ntotal} vectors")
    
    # Save index
    index_path = os.path.join(args.output_dir, 'faiss_index_gt_reports.bin')
    faiss.write_index(index, index_path)
    print(f"  ✓ Index saved: {index_path}")
    
    # --- Build train cache ---
    print(f"\n[4/6] Building train retrieval cache...")
    
    # Convert drafts dict to ordered list
    train_draft_texts = [train_drafts.get(i, "") for i in range(len(train_df))]
    
    # Handle empty drafts
    n_empty = sum(1 for t in train_draft_texts if not t.strip())
    if n_empty > 0:
        print(f"  WARNING: {n_empty} empty train drafts (will get random retrievals)")
    
    # Encode train drafts
    train_draft_embeddings = encode_texts(
        train_draft_texts, encoder,
        batch_size=args.batch_size, desc="Encoding train drafts"
    )
    
    # Retrieve (exclude self for training set)
    self_indices = list(range(len(train_df)))
    train_cache = retrieve_for_drafts(
        train_draft_embeddings, index, gt_findings_train,
        k=args.k, self_indices=self_indices
    )
    
    train_cache_path = os.path.join(args.output_dir, 'retrieval_cache_train_gtr.pkl')
    with open(train_cache_path, 'wb') as f:
        pickle.dump(train_cache, f)
    print(f"  ✓ Saved train cache: {train_cache_path}")
    
    analyze_retrieval_quality(train_cache, gt_findings_train, train_drafts)
    
    # --- Build val cache ---
    if val_drafts:
        print(f"\n[5/6] Building val retrieval cache...")
        
        val_df = pd.read_csv(args.val_csv)
        val_draft_texts = [val_drafts.get(i, "") for i in range(len(val_df))]
        
        val_draft_embeddings = encode_texts(
            val_draft_texts, encoder,
            batch_size=args.batch_size, desc="Encoding val drafts"
        )
        
        # No self-exclusion for val (querying against training index)
        val_cache = retrieve_for_drafts(
            val_draft_embeddings, index, gt_findings_train,
            k=args.k, self_indices=None
        )
        
        val_cache_path = os.path.join(args.output_dir, 'retrieval_cache_val_gtr.pkl')
        with open(val_cache_path, 'wb') as f:
            pickle.dump(val_cache, f)
        print(f"  ✓ Saved val cache: {val_cache_path}")
        
        analyze_retrieval_quality(val_cache, gt_findings_train, val_drafts)
    else:
        print(f"\n[5/6] Skipping val cache (no val drafts provided)")
    
    # --- Summary ---
    print(f"\n[6/6] Summary")
    print("=" * 80)
    print(f"✅ GENERATE-THEN-RETRIEVE CACHES COMPLETE")
    print("=" * 80)
    print(f"  FAISS index:     {index_path}")
    print(f"  Train cache:     {train_cache_path}")
    if val_drafts:
        print(f"  Val cache:       {val_cache_path}")
    print(f"  k:               {args.k}")
    print(f"  Encoder:         {args.encoder_model}")
    print()
    print("Next step: Retrain Stage 2 with these caches:")
    print(f"  python train_stage2_cached.py \\")
    print(f"    --stage1_checkpoint ./checkpoints/stage1_opt_clean/best_model.pt \\")
    print(f"    --train_csv {args.train_csv} \\")
    if args.val_csv:
        print(f"    --val_csv {args.val_csv} \\")
    print(f"    --train_cache {train_cache_path} \\")
    if val_drafts:
        print(f"    --val_cache {val_cache_path} \\")
    print(f"    --output_dir ./checkpoints/stage2_gen_then_retrieve")
    print("=" * 80)


if __name__ == '__main__':
    main()
