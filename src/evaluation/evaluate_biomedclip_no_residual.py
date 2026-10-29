"""
BiomedCLIP Retrieval Experiment for WACV Paper

Tests whether a biomedically-trained vision encoder (BiomedCLIP) can close
the cross-modal retrieval gap compared to the general ViT encoder used in
the original image retrieval approach.

Pipeline:
  1. Encode all training images with BiomedCLIP → build FAISS index
  2. For each val image, encode with BiomedCLIP → retrieve top-3 similar training reports
  3. Feed val image + retrieved reports into Stage 2 model → generate report
  4. Compute BLEU/ROUGE/METEOR metrics

This is a DROP-IN replacement for the image retrieval condition.
Same Stage 2 model, same k=3, same evaluation — only the retrieval encoder changes.

Usage:
    # Step 1: Build the BiomedCLIP index (~1-2 hours)
    python evaluate_biomedclip_retrieval.py \
        --mode build_index \
        --train_csv ./data/processed/mimic_cxr_train_clean.csv \
        --output_dir ./data/biomedclip_retrieval \
        --device cuda:0 \
        --batch_size 64

    # Step 2: Evaluate Stage 2 with BiomedCLIP retrieval (~1 hour)
    python evaluate_biomedclip_retrieval.py \
        --mode evaluate \
        --checkpoint ./checkpoints/stage2_gen_then_retrieve/best_model.pt \
        --csv_file ./data/processed/mimic_cxr_val_clean.csv \
        --index_dir ./data/biomedclip_retrieval \
        --output_dir ./results/stage2_biomedclip_retrieval \
        --device cuda:0 \
        --batch_size 8 \
        --max_samples 1000

    # Or do both in one go:
    python evaluate_biomedclip_retrieval.py \
        --mode both \
        --train_csv ./data/processed/mimic_cxr_train_clean.csv \
        --checkpoint ./checkpoints/stage2_gen_then_retrieve/best_model.pt \
        --csv_file ./data/processed/mimic_cxr_val_clean.csv \
        --output_dir ./results/stage2_biomedclip_retrieval \
        --device cuda:0
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
import os
import sys
import json
import torch
import argparse
import numpy as np
import pandas as pd
import faiss
import pickle
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader



# =====================================================
# BIOMEDCLIP ENCODER
# =====================================================

class BiomedCLIPEncoder:
    """Wrapper for BiomedCLIP image encoding"""

    def __init__(self, device='cuda:0'):
        from open_clip import create_model_and_transforms, get_tokenizer

        self.device = device
        print("Loading BiomedCLIP model...")

        self.model, self.preprocess_train, self.preprocess_val = create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.model = self.model.to(device)
        self.model.eval()

        # Get embedding dimension
        # BiomedCLIP uses 512-dim embeddings
        self.embed_dim = 512
        print(f"  BiomedCLIP loaded. Embedding dim: {self.embed_dim}")

    def encode_images(self, images):
        """Encode a batch of PIL images to embeddings.
        
        Args:
            images: list of PIL Images
        Returns:
            numpy array of shape [batch, embed_dim]
        """
        processed = torch.stack([self.preprocess_val(img) for img in images]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(processed)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().astype('float32')

    def encode_single(self, image):
        """Encode a single PIL image"""
        return self.encode_images([image])


# =====================================================
# DATASET FOR INDEX BUILDING
# =====================================================

class SimpleImageDataset(Dataset):
    """Load images for BiomedCLIP encoding"""

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        findings = str(row['findings'])

        try:
            image = Image.open(image_path).convert('RGB')
            valid = True
        except Exception:
            image = Image.new('RGB', (224, 224))
            valid = False

        return {
            'image': image,
            'findings': findings,
            'image_path': image_path,
            'idx': idx,
            'valid': valid,
        }


def collate_images(batch):
    """Custom collate that keeps PIL images as a list"""
    return {
        'images': [b['image'] for b in batch],
        'findings': [b['findings'] for b in batch],
        'image_paths': [b['image_path'] for b in batch],
        'indices': [b['idx'] for b in batch],
        'valid': [b['valid'] for b in batch],
    }


# =====================================================
# STEP 1: BUILD BIOMEDCLIP INDEX
# =====================================================

def build_biomedclip_index(train_csv, output_dir, device='cuda:0', batch_size=64):
    """Build FAISS index using BiomedCLIP image embeddings"""

    print("=" * 80)
    print("BUILDING BIOMEDCLIP IMAGE RETRIEVAL INDEX")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Load BiomedCLIP
    encoder = BiomedCLIPEncoder(device=device)

    # Load dataset
    print(f"\nLoading training data: {train_csv}")
    dataset = SimpleImageDataset(train_csv)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_images,
        pin_memory=False,
    )
    print(f"  Dataset: {len(dataset)} samples")

    # Extract embeddings
    print("\nExtracting BiomedCLIP embeddings...")
    all_embeddings = []
    all_findings = []
    all_image_paths = []
    all_indices = []
    invalid_count = 0

    for batch in tqdm(dataloader, desc="Encoding"):
        images = batch['images']
        embeddings = encoder.encode_images(images)

        all_embeddings.append(embeddings)
        all_findings.extend(batch['findings'])
        all_image_paths.extend(batch['image_paths'])
        all_indices.extend(batch['indices'])
        invalid_count += sum(1 for v in batch['valid'] if not v)

    embeddings_matrix = np.vstack(all_embeddings)
    print(f"\n  Embeddings shape: {embeddings_matrix.shape}")
    print(f"  Invalid images (used dummy): {invalid_count}")

    # Build FAISS index
    print("\nBuilding FAISS index...")
    dimension = embeddings_matrix.shape[1]

    # Use IndexFlatIP for cosine similarity (embeddings are already L2-normalized)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)

    print(f"  FAISS index: {index.ntotal} vectors, dim={dimension}")

    # Save everything
    index_path = os.path.join(output_dir, 'biomedclip_index.faiss')
    faiss.write_index(index, index_path)

    embeddings_path = os.path.join(output_dir, 'biomedclip_embeddings.npy')
    np.save(embeddings_path, embeddings_matrix)

    metadata = {
        'findings': all_findings,
        'image_paths': all_image_paths,
        'indices': all_indices,
        'dimension': dimension,
        'count': len(all_findings),
        'encoder': 'BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        'invalid_images': invalid_count,
    }
    metadata_path = os.path.join(output_dir, 'biomedclip_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n  Saved: {index_path} ({os.path.getsize(index_path)/1e6:.0f} MB)")
    print(f"  Saved: {embeddings_path} ({os.path.getsize(embeddings_path)/1e6:.0f} MB)")
    print(f"  Saved: {metadata_path} ({os.path.getsize(metadata_path)/1e6:.0f} MB)")

    print("\n" + "=" * 80)
    print("BIOMEDCLIP INDEX COMPLETE")
    print("=" * 80)

    return index, metadata


# =====================================================
# STEP 2: EVALUATE WITH BIOMEDCLIP RETRIEVAL
# =====================================================

class BiomedCLIPRetrievalSystem:
    """Retrieval using BiomedCLIP embeddings"""

    def __init__(self, index_dir):
        index_path = os.path.join(index_dir, 'biomedclip_index.faiss')
        metadata_path = os.path.join(index_dir, 'biomedclip_metadata.pkl')

        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        print(f"  Loaded BiomedCLIP retrieval: {self.index.ntotal} vectors, dim={self.metadata['dimension']}")

    def retrieve(self, image_embedding, k=3):
        """Retrieve top-k reports given a BiomedCLIP image embedding"""
        if len(image_embedding.shape) == 1:
            image_embedding = image_embedding[np.newaxis, :]

        # Embeddings are already normalized from encoding step
        distances, indices = self.index.search(image_embedding.astype('float32'), k)

        retrieved_texts = [self.metadata['findings'][idx] for idx in indices[0]]
        return retrieved_texts


def evaluate_with_biomedclip(checkpoint, csv_file, index_dir, output_dir,
                              device='cuda:0', batch_size=8, max_samples=None, k=3):
    """Run Stage 2 evaluation with BiomedCLIP-based retrieval"""

    print("=" * 80)
    print("EVALUATION WITH BIOMEDCLIP IMAGE RETRIEVAL")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    device_obj = torch.device(device)

    # Load Stage 2 model
    print("\n[1/4] Loading Stage 2 model...")
    from modified_qformer_opt_no_residual import RAGEnhancedBLIP2_OPT

    model = RAGEnhancedBLIP2_OPT(
        blip2_model_name='Salesforce/blip2-opt-2.7b',
        num_query_tokens=32,
        freeze_vision=True,
        freeze_llm=True,
        use_retrieval=True,
    )

    ckpt = torch.load(checkpoint, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device_obj)
    model.eval()
    print("  Stage 2 model loaded")

    # Load BiomedCLIP retrieval
    print("\n[2/4] Loading BiomedCLIP retrieval system...")
    retrieval = BiomedCLIPRetrievalSystem(index_dir)

    # Load BiomedCLIP encoder for val image encoding
    print("\n  Loading BiomedCLIP encoder for val queries...")
    encoder = BiomedCLIPEncoder(device=device)

    # Load dataset
    print("\n[3/4] Loading val dataset...")
    from dataset_opt import MIMICCXRDataset, collate_fn as base_collate_fn

    dataset = MIMICCXRDataset(
        csv_file=csv_file,
        tokenizer_name='Salesforce/blip2-opt-2.7b',
        max_length=128,
    )

    if max_samples:
        from torch.utils.data import Subset
        indices = list(range(min(max_samples, len(dataset))))
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=base_collate_fn,
    )
    print(f"  Val dataset: {len(dataset)} samples")

    # Also load the CSV directly to read image_paths for BiomedCLIP encoding
    df = pd.read_csv(csv_file)
    if max_samples:
        df = df.head(max_samples)

    # Generation config — match the original evaluation
    generation_config = {
        'max_length': 100,
        'num_beams': 5,
        'early_stopping': True,
        'no_repeat_ngram_size': 3,
        'length_penalty': 1.0,
    }

    # Generate predictions
    print("\n[4/4] Generating with BiomedCLIP retrieval...")

    predictions = []
    references = []
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            try:
                pixel_values = batch['pixel_values'].to(device_obj)
                current_batch_size = pixel_values.shape[0]

                # Step 1: Encode batch images with BiomedCLIP for retrieval
                batch_images = []
                for i in range(current_batch_size):
                    img_path = df.iloc[sample_idx + i]['image_path']
                    try:
                        img = Image.open(img_path).convert('RGB')
                    except Exception:
                        img = Image.new('RGB', (224, 224))
                    batch_images.append(img)

                biomedclip_embeddings = encoder.encode_images(batch_images)

                # Step 2: Retrieve top-k reports using BiomedCLIP embeddings
                batch_retrieved_texts = []
                for emb in biomedclip_embeddings:
                    retrieved = retrieval.retrieve(emb, k=k)
                    batch_retrieved_texts.append(retrieved)

                # Step 3: Generate with Stage 2 model (image + retrieved context)
                outputs = model.generate(
                    pixel_values=pixel_values,
                    retrieved_texts=batch_retrieved_texts,
                    **generation_config,
                )

                preds = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                refs = batch.get('text', [''] * current_batch_size)

                predictions.extend(preds)
                references.extend(refs if refs else [''] * current_batch_size)

                sample_idx += current_batch_size

            except Exception as e:
                print(f"\n  Error at sample {sample_idx}: {e}")
                current_batch_size = pixel_values.shape[0]
                predictions.extend([''] * current_batch_size)
                references.extend([''] * current_batch_size)
                sample_idx += current_batch_size
                continue

    print(f"\n  Generated {len(predictions)} predictions")

    # Save predictions
    results = {
        'predictions': predictions,
        'references': references,
        'method': 'biomedclip_image_retrieval',
        'retrieval_k': k,
        'checkpoint': str(checkpoint),
        'encoder': 'BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
    }

    pred_file = os.path.join(output_dir, 'predictions.json')
    with open(pred_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Compute metrics
    print("\n  Computing metrics...")

    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    from rouge_score import rouge_scorer
    import nltk

    for pkg, name in [('tokenizers/punkt', 'punkt'), ('tokenizers/punkt_tab', 'punkt_tab'), ('corpora/wordnet', 'wordnet')]:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(name, quiet=True)

    # Filter valid pairs
    valid = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    vp = [v[0] for v in valid]
    vr = [v[1] for v in valid]

    if not valid:
        print("  ERROR: No valid predictions!")
        return {}

    print(f"  Valid samples: {len(valid)}/{len(predictions)}")

    # BLEU
    refs_tok = [[word_tokenize(r.lower())] for r in vr]
    preds_tok = [word_tokenize(p.lower()) for p in vp]
    smooth = SmoothingFunction().method1

    metrics = {
        'n_valid': len(valid),
        'n_total': len(predictions),
        'bleu1': corpus_bleu(refs_tok, preds_tok, weights=(1, 0, 0, 0), smoothing_function=smooth),
        'bleu4': corpus_bleu(refs_tok, preds_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth),
    }

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    metrics['rouge_l'] = float(np.mean([scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(vp, vr)]))

    # METEOR
    try:
        from nltk.translate.meteor_score import meteor_score
        metrics['meteor'] = float(np.mean([meteor_score([word_tokenize(r.lower())], word_tokenize(p.lower())) for p, r in zip(vp, vr)]))
    except Exception:
        pass

    # Save metrics
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print results
    print(f"\n{'=' * 80}")
    print("RESULTS: BIOMEDCLIP IMAGE RETRIEVAL")
    print(f"{'=' * 80}")
    for k_name, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k_name:15s}: {v:.4f}")
        else:
            print(f"  {k_name:15s}: {v}")
    print(f"{'=' * 80}")

    print(f"\n  Next: Run RadGraph F1:")
    print(f"  python evaluate_radgraph_final.py --predictions_file {pred_file} --output_dir {output_dir}")

    return metrics


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(description='BiomedCLIP Retrieval Experiment')
    parser.add_argument('--mode', type=str, choices=['build_index', 'evaluate', 'both'],
                        default='both', help='Which step to run')

    # Index building args
    parser.add_argument('--train_csv', type=str,
                        default='./data/processed/mimic_cxr_train_clean.csv')

    # Evaluation args
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/stage2_gen_then_retrieve/best_model.pt')
    parser.add_argument('--csv_file', type=str,
                        default='./data/processed/mimic_cxr_val_clean.csv')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--k', type=int, default=3, help='Number of reports to retrieve')

    # Shared args
    parser.add_argument('--index_dir', type=str, default='./data/biomedclip_retrieval',
                        help='Directory for BiomedCLIP index (output for build, input for evaluate)')
    parser.add_argument('--output_dir', type=str, default='./results/stage2_biomedclip_retrieval')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (64 for index building, 8 for eval)')

    args = parser.parse_args()

    if args.mode in ('build_index', 'both'):
        build_biomedclip_index(
            train_csv=args.train_csv,
            output_dir=args.index_dir,
            device=args.device,
            batch_size=args.batch_size,
        )

    if args.mode in ('evaluate', 'both'):
        evaluate_with_biomedclip(
            checkpoint=args.checkpoint,
            csv_file=args.csv_file,
            index_dir=args.index_dir,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=8,  # Smaller for eval (Stage 2 model needs more memory)
            max_samples=args.max_samples,
            k=args.k,
        )


if __name__ == '__main__':
    main()
