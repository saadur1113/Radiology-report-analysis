"""
Build relevant retrieval cache for Stage 1 training.
Uses BiomedCLIP image embeddings to find similar training
images and retrieve their ground-truth reports.

Usage:
    CUDA_VISIBLE_DEVICES=6 python build_stage1_biomedclip_cache.py \
        --train_csv ./data/processed/mimic_cxr_train_clean.csv \
        --biomedclip_index_dir ./data/biomedclip_retrieval \
        --output_path ./data/stage1_biomedclip_cache/retrieval_cache_train.pkl \
        --k 3 --batch_size 32 --device cuda:0
"""

import torch
import numpy as np
import pandas as pd
import pickle
import faiss
from tqdm import tqdm
from PIL import Image
import argparse
import os


def build_stage1_retrieval_cache(
    train_csv,
    biomedclip_index_dir,
    output_path,
    k=3,
    batch_size=32,
    device='cuda:0'
):
    """Build retrieval cache using BiomedCLIP for Stage 1 training."""
    
    # Load training data
    df = pd.read_csv(train_csv)
    print(f"Loaded {len(df)} training samples")
    
    # Check for existing FAISS index
    index_path = os.path.join(biomedclip_index_dir, 'biomedclip_index.faiss')
    embeddings_path = os.path.join(biomedclip_index_dir, 'biomedclip_embeddings.npy')
    
    if os.path.exists(index_path):
        print(f"Loading existing FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        print(f"  Index size: {index.ntotal}")
        
        # Load or rebuild embeddings for query
        if os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path)
            print(f"  Loaded embeddings: {embeddings.shape}")
        else:
            print("  Need to rebuild embeddings for querying...")
            embeddings = reconstruct_embeddings(index)
    else:
        print(f"No existing index at {index_path}")
        print("Building BiomedCLIP embeddings from scratch...")
        embeddings, index = build_index_from_scratch(
            df, biomedclip_index_dir, batch_size, device
        )
    
    # For each training sample, find k+1 nearest neighbors
    # (k+1 because the closest match is the image itself)
    print(f"\nRetrieving top-{k} neighbors for each training sample...")
    
    reports = df['findings'].astype(str).tolist()
    retrieval_cache = {}
    
    # Search in batches
    search_batch = 512
    for i in tqdm(range(0, len(df), search_batch), desc="Searching"):
        batch_end = min(i + search_batch, len(df))
        query_embeddings = embeddings[i:batch_end].astype('float32')
        
        # Search for k+1 (first result is self)
        distances, indices = index.search(query_embeddings, k + 1)
        
        for j in range(batch_end - i):
            sample_idx = i + j
            # Skip self (first result is always the query itself)
            neighbor_indices = [
                int(idx) for idx in indices[j]
                if idx != sample_idx and idx >= 0
            ][:k]
            
            # Get their reports
            retrieved_reports = [reports[idx] for idx in neighbor_indices]
            retrieval_cache[sample_idx] = retrieved_reports
    
    # Save cache
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\nSaving cache to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(retrieval_cache, f)
    
    # Verify
    print(f"\nCache statistics:")
    print(f"  Total samples: {len(retrieval_cache)}")
    print(f"  Expected: {len(df)}")
    
    # Check a few samples
    for idx in [0, 100, 1000]:
        if idx in retrieval_cache:
            reports_list = retrieval_cache[idx]
            print(f"  Sample {idx}: {len(reports_list)} reports, "
                  f"first report: {reports_list[0][:60]}...")
    
    return retrieval_cache


def build_index_from_scratch(df, output_dir, batch_size, device):
    """Build BiomedCLIP embeddings and FAISS index from scratch."""
    from open_clip import create_model_and_transforms
    
    model, _, preprocess = create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model = model.to(device).eval()
    
    embeddings = []
    for i in tqdm(range(0, len(df), batch_size), desc="Encoding images"):
        batch_paths = df.iloc[i:i+batch_size]['image_path'].tolist()
        batch_images = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                img = preprocess(img)
                batch_images.append(img)
            except Exception:
                batch_images.append(torch.zeros(3, 224, 224))
        
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            features = model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy())
    
    embeddings = np.vstack(embeddings).astype('float32')
    
    # Build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, 'biomedclip_index.faiss')
    embeddings_path = os.path.join(output_dir, 'biomedclip_embeddings.npy')
    faiss.write_index(index, index_path)
    np.save(embeddings_path, embeddings)
    print(f"Saved index ({index.ntotal} vectors) to {index_path}")
    
    return embeddings, index


def reconstruct_embeddings(index):
    """Reconstruct embeddings from FAISS index."""
    n = index.ntotal
    dim = index.d
    embeddings = np.zeros((n, dim), dtype='float32')
    for i in range(n):
        embeddings[i] = index.reconstruct(i)
    return embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build BiomedCLIP retrieval cache for Stage 1 training'
    )
    parser.add_argument('--train_csv', type=str,
        default='./data/processed/mimic_cxr_train_clean.csv')
    parser.add_argument('--biomedclip_index_dir', type=str,
        default='./data/biomedclip_retrieval')
    parser.add_argument('--output_path', type=str,
        default='./data/stage1_biomedclip_cache/retrieval_cache_train.pkl')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    build_stage1_retrieval_cache(
        train_csv=args.train_csv,
        biomedclip_index_dir=args.biomedclip_index_dir,
        output_path=args.output_path,
        k=args.k,
        batch_size=args.batch_size,
        device=args.device
    )
