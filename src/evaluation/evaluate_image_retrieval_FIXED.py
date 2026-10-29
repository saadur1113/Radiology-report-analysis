"""
Modified Evaluation with IMAGE-BASED Retrieval

This version bypasses the dataset's GT-text retrieval and uses
image embeddings for retrieval queries instead.

Usage:
    python evaluate_image_retrieval_FIXED.py \
        --checkpoint ./checkpoints/stage2_opt_clean/best_model.pt \
        --csv_file ./data/processed/mimic_cxr_val_clean.csv \
        --image_index_dir ./data/image_retrieval \
        --output_dir ./results/stage2_image_retrieval \
        --max_samples 1000 \
        --batch_size 8
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import faiss
import pickle
from pathlib import Path


from modified_qformer_opt import RAGEnhancedBLIP2_OPT
from dataset_opt import MIMICCXRDataset, collate_fn  # Use base dataset WITHOUT retrieval
from torch.utils.data import DataLoader


class ImageRetrievalSystem:
    """Image-based retrieval (not text-based)"""
    def __init__(self, index_dir):
        index_path = os.path.join(index_dir, 'image_index.faiss')
        self.index = faiss.read_index(index_path)
        
        metadata_path = os.path.join(index_dir, 'image_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"✓ Loaded image retrieval: {self.index.ntotal} vectors")
    
    def retrieve(self, image_embedding, k=3):
        """Retrieve using IMAGE embedding"""
        if len(image_embedding.shape) == 1:
            image_embedding = image_embedding[np.newaxis, :]
        
        faiss.normalize_L2(image_embedding)
        distances, indices = self.index.search(image_embedding.astype('float32'), k)
        
        retrieved_texts = [self.metadata['findings'][idx] for idx in indices[0]]
        return retrieved_texts


def evaluate_with_image_retrieval(args):
    """Evaluation using IMAGE-based retrieval"""
    
    print("="*80)
    print("EVALUATION WITH IMAGE-BASED RETRIEVAL")
    print("="*80)
    
    device = torch.device(args.device)
    
    # Load model
    print("\n[1/4] Loading model...")
    model = RAGEnhancedBLIP2_OPT(
        blip2_model_name='Salesforce/blip2-opt-2.7b',
        num_query_tokens=32,
        freeze_vision=True,
        freeze_llm=True,
        use_retrieval=True
    )
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Load image retrieval system
    print("\n[2/4] Loading image retrieval...")
    retrieval = ImageRetrievalSystem(args.image_index_dir)
    
    # Load dataset (WITHOUT retrieval - we'll do it manually)
    print("\n[3/4] Loading dataset...")
    dataset = MIMICCXRDataset(
        csv_file=args.csv_file,
        tokenizer_name='Salesforce/blip2-opt-2.7b',
        max_length=128
    )
    
    if args.max_samples:
        from torch.utils.data import Subset
        indices = list(range(min(args.max_samples, len(dataset))))
        dataset = Subset(dataset, indices)
    
    print(f"✓ Dataset: {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate predictions
    print("\n[4/4] Generating with IMAGE-based retrieval...")
    
    predictions = []
    references = []
    
    generation_config = {
        'max_length': 100,
        'num_beams': 5,
        'early_stopping': True,
        'no_repeat_ngram_size': 3,
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            try:
                pixel_values = batch['pixel_values'].to(device)
                batch_size = pixel_values.shape[0]
                
                # Step 1: Get image embeddings for retrieval
                vision_outputs = model.vision_model(pixel_values, return_dict=True)
                vision_features = vision_outputs.last_hidden_state
                
                if hasattr(model, 'vision_projection') and model.vision_projection is not None:
                    vision_features = model.vision_projection(vision_features)
                
                # Pool to single vector per image
                image_embeds = vision_features.mean(dim=1).cpu().numpy()
                
                # Step 2: Retrieve using IMAGE embeddings (not GT text!)
                batch_retrieved_texts = []
                for img_emb in image_embeds:
                    retrieved = retrieval.retrieve(img_emb, k=3)
                    batch_retrieved_texts.append(retrieved)
                
                # Step 3: Generate with retrieved context
                outputs = model.generate(
                    pixel_values=pixel_values,
                    retrieved_texts=batch_retrieved_texts,
                    **generation_config
                )
                
                # Decode
                preds = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                refs = batch.get('text', [''] * batch_size)
                
                predictions.extend(preds)
                references.extend(refs)
                
            except Exception as e:
                print(f"\nError in batch: {e}")
                # Add empty predictions to maintain alignment
                predictions.extend([''] * batch_size)
                references.extend([''] * batch_size)
                continue
    
    print(f"\n✓ Generated {len(predictions)} predictions")
    
    # Save predictions
    results = {
        'predictions': predictions,
        'references': references,
        'method': 'image_based_retrieval'
    }
    
    predictions_file = os.path.join(args.output_dir, 'predictions.json')
    with open(predictions_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved: {predictions_file}")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    from rouge_score import rouge_scorer
    from nltk.translate.meteor_score import meteor_score
    import nltk
    
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except:
        nltk.download('wordnet', quiet=True)
    
    # BLEU
    refs_tok = [[word_tokenize(r.lower())] for r in references]
    preds_tok = [word_tokenize(p.lower()) for p in predictions]
    smoothing = SmoothingFunction().method1
    
    bleu1 = corpus_bleu(refs_tok, preds_tok, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(refs_tok, preds_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(refs_tok, preds_tok, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(refs_tok, preds_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(r, p)['rougeL'].fmeasure for r, p in zip(references, predictions)]
    rouge_l = sum(rouge_scores) / len(rouge_scores)
    
    # METEOR
    meteor_scores = [meteor_score([word_tokenize(r.lower())], word_tokenize(p.lower())) 
                     for r, p in zip(references, predictions)]
    meteor = sum(meteor_scores) / len(meteor_scores)
    
    metrics = {
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu3': bleu3,
        'bleu4': bleu4,
        'rouge_l': rouge_l,
        'meteor': meteor
    }
    
    metrics_file = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Saved: {metrics_file}")
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS (IMAGE-BASED RETRIEVAL - HONEST)")
    print("="*80)
    print(f"BLEU-1:  {bleu1:.4f}")
    print(f"BLEU-4:  {bleu4:.4f}")
    print(f"ROUGE-L: {rouge_l:.4f}")
    print(f"METEOR:  {meteor:.4f}")
    print("="*80)
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--csv_file', required=True)
    parser.add_argument('--image_index_dir', required=True,
                       help='Directory with image_index.faiss and image_metadata.pkl')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    
    args = parser.parse_args()
    
    evaluate_with_image_retrieval(args)
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()