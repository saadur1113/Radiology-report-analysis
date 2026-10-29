"""
Build Image-Based FAISS Retrieval Index

This creates a retrieval index based on IMAGE embeddings (not text),
which is the correct approach for realistic evaluation.

Usage:
    python build_image_retrieval_index.py \
        --stage1_model ./checkpoints/stage1_opt_clean/best_model.pt \
        --train_csv ./data/processed/mimic_cxr_train_clean.csv \
        --output_dir ./data/image_retrieval
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import faiss
import pickle
import argparse
import os
from tqdm import tqdm
import sys


from modified_qformer_opt import RAGEnhancedBLIP2_OPT


class ImageDataset(Dataset):
    """Simple dataset to load images"""
    def __init__(self, csv_file, image_processor):
        self.data = pd.read_csv(csv_file)
        self.processor = image_processor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        findings = row['findings']
        
        try:
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.processor(image, return_tensors="pt")['pixel_values'][0]
            return {
                'pixel_values': pixel_values,
                'findings': findings,
                'image_path': image_path,
                'idx': idx
            }
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return dummy data
            dummy_image = Image.new('RGB', (224, 224))
            pixel_values = self.processor(dummy_image, return_tensors="pt")['pixel_values'][0]
            return {
                'pixel_values': pixel_values,
                'findings': findings,
                'image_path': image_path,
                'idx': idx
            }


def build_image_index(model_path, train_csv, output_dir, batch_size=32):
    """
    Build FAISS index using image embeddings from Stage 1 model
    """
    
    print("="*80)
    print("BUILDING IMAGE-BASED RETRIEVAL INDEX")
    print("="*80)
    
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Stage 1 model (has vision encoder)
    print(f"\nLoading Stage 1 model: {model_path}")
    model = RAGEnhancedBLIP2_OPT(
        blip2_model_name='Salesforce/blip2-opt-2.7b',
        num_query_tokens=32,
        freeze_vision=True,
        freeze_llm=True,
        use_retrieval=False  # Don't need retrieval components
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded")
    
    # Get image processor from model
    from transformers import BlipImageProcessor
    processor = BlipImageProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
    
    # Load dataset
    print(f"\nLoading training data: {train_csv}")
    dataset = ImageDataset(train_csv, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ Dataset: {len(dataset)} samples")
    
    # Extract image embeddings
    print("\nExtracting image embeddings...")
    
    all_embeddings = []
    all_findings = []
    all_image_paths = []
    all_indices = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing"):
            pixel_values = batch['pixel_values'].to(device)
            
            # Get vision features from model
            vision_outputs = model.vision_model(pixel_values, return_dict=True)
            image_features = vision_outputs.last_hidden_state
            
            # Project if needed
            if model.vision_projection is not None:
                image_features = model.vision_projection(image_features)
            
            # Pool to single vector per image (mean pooling)
            # Shape: [batch, seq_len, hidden_dim] -> [batch, hidden_dim]
            image_embeddings = image_features.mean(dim=1)
            
            # Store
            all_embeddings.append(image_embeddings.cpu().numpy())
            all_findings.extend(batch['findings'])
            all_image_paths.extend(batch['image_path'])
            all_indices.extend(batch['idx'].tolist())
    
    # Concatenate all embeddings
    embeddings_matrix = np.vstack(all_embeddings)
    
    print(f"\n✓ Extracted embeddings: {embeddings_matrix.shape}")
    print(f"  Dimension: {embeddings_matrix.shape[1]}")
    print(f"  Count: {embeddings_matrix.shape[0]}")
    
    # Build FAISS index
    print("\nBuilding FAISS index...")
    dimension = embeddings_matrix.shape[1]
    
    # Use L2 distance (Euclidean)
    index = faiss.IndexFlatL2(dimension)
    
    # Normalize vectors for cosine similarity (optional but recommended)
    faiss.normalize_L2(embeddings_matrix)
    
    # Add vectors to index
    index.add(embeddings_matrix.astype('float32'))
    
    print(f"✓ FAISS index built with {index.ntotal} vectors")
    
    # Save index
    index_path = os.path.join(output_dir, 'image_index.faiss')
    faiss.write_index(index, index_path)
    print(f"✓ Saved index: {index_path}")
    
    # Save metadata (findings texts, image paths, original indices)
    metadata = {
        'findings': all_findings,
        'image_paths': all_image_paths,
        'indices': all_indices,
        'dimension': dimension,
        'count': len(all_findings)
    }
    
    metadata_path = os.path.join(output_dir, 'image_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✓ Saved metadata: {metadata_path}")
    
    # Save embeddings (optional, for debugging)
    embeddings_path = os.path.join(output_dir, 'image_embeddings.npy')
    np.save(embeddings_path, embeddings_matrix)
    print(f"✓ Saved embeddings: {embeddings_path}")
    
    # Summary
    index_size_mb = os.path.getsize(index_path) / (1024 * 1024)
    metadata_size_mb = os.path.getsize(metadata_path) / (1024 * 1024)
    
    print("\n" + "="*80)
    print("✅ IMAGE-BASED RETRIEVAL INDEX COMPLETE")
    print("="*80)
    print(f"Index size: {index_size_mb:.1f} MB")
    print(f"Metadata size: {metadata_size_mb:.1f} MB")
    print(f"Total vectors: {index.ntotal}")
    print(f"Dimension: {dimension}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    return index, metadata


def test_retrieval(index_path, metadata_path, test_query_idx=0):
    """Test the retrieval system"""
    
    print("\nTesting retrieval...")
    
    # Load index
    index = faiss.read_index(index_path)
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Load embeddings
    embeddings = np.load(index_path.replace('image_index.faiss', 'image_embeddings.npy'))
    
    # Test query
    query_emb = embeddings[test_query_idx:test_query_idx+1]
    
    # Search
    D, I = index.search(query_emb, k=4)  # k=4 to see query itself + 3 neighbors
    
    print(f"\nQuery (index {test_query_idx}):")
    print(f"  Image: {metadata['image_paths'][test_query_idx]}")
    print(f"  Findings: {metadata['findings'][test_query_idx][:100]}...")
    
    print(f"\nTop 4 Retrieved:")
    for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
        print(f"\n  Rank {rank+1} (distance: {dist:.4f}):")
        print(f"    Index: {idx}")
        print(f"    Image: {metadata['image_paths'][idx]}")
        print(f"    Findings: {metadata['findings'][idx][:100]}...")
    
    print("\n✓ Retrieval test complete")


def main():
    parser = argparse.ArgumentParser(description='Build image-based retrieval index')
    parser.add_argument('--stage1_model', type=str, required=True,
                       help='Path to Stage 1 model checkpoint')
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to training CSV')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for index and metadata')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--test', action='store_true',
                       help='Run test retrieval after building')
    
    args = parser.parse_args()
    
    # Build index
    index, metadata = build_image_index(
        model_path=args.stage1_model,
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    # Test if requested
    if args.test:
        index_path = os.path.join(args.output_dir, 'image_index.faiss')
        metadata_path = os.path.join(args.output_dir, 'image_metadata.pkl')
        test_retrieval(index_path, metadata_path)


if __name__ == '__main__':
    main()
