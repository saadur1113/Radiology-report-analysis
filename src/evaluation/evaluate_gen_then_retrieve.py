"""
Evaluate Stage 2 Gen-Then-Retrieve Model

Uses the generate-then-retrieve val cache for retrieval context,
matching the realistic deployment scenario.

Usage:
    python evaluate_gen_then_retrieve.py \
        --checkpoint ./checkpoints/stage2_gen_then_retrieve/checkpoint_epoch_2.pt \
        --csv_file ./data/processed/mimic_cxr_val_clean.csv \
        --val_cache ./data/gen_then_retrieve/retrieval_cache_val_gtr.pkl \
        --output_dir ./results/stage2_gen_then_retrieve \
        --batch_size 8 \
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
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path


from modified_qformer_opt import RAGEnhancedBLIP2_OPT
from dataset_opt_cached import MIMICCXRDatasetCached, collate_fn
from torch.utils.data import DataLoader


def evaluate(checkpoint_path, csv_file, val_cache, output_dir, batch_size=8, device='cuda:0'):
    
    print("=" * 80)
    print("EVALUATE: STAGE 2 GENERATE-THEN-RETRIEVE")
    print("=" * 80)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  CSV:        {csv_file}")
    print(f"  Val cache:  {val_cache}")
    print(f"  Output:     {output_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device:     {device}")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # --- Load Model ---
    print("\n[1/4] Loading model...")
    model = RAGEnhancedBLIP2_OPT(
        blip2_model_name='Salesforce/blip2-opt-2.7b',
        num_query_tokens=32,
        freeze_vision=True,
        freeze_llm=True,
        use_retrieval=True
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', '?')
        val_loss = checkpoint.get('val_loss', '?')
        print(f"  Checkpoint epoch: {epoch}, val_loss: {val_loss}")
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("  Model loaded")
    
    # --- Load Dataset with Cached Retrieval ---
    print("\n[2/4] Loading dataset with gen-then-retrieve cache...")
    dataset = MIMICCXRDatasetCached(
        csv_file=csv_file,
        retrieval_cache_file=val_cache,
        tokenizer_name='Salesforce/blip2-opt-2.7b',
        max_length=128
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"  Dataset: {len(dataset)} samples, {len(dataloader)} batches")
    
    # --- Generate ---
    print("\n[3/4] Generating reports...")
    
    generation_config = {
    'max_length': 100,
    'num_beams': 5,
    'early_stopping': True,
    'no_repeat_ngram_size': 3,
    'length_penalty': 1.0,
}
    
    all_predictions = []
    all_references = []
    errors = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            try:
                pixel_values = batch['pixel_values'].to(device)
                retrieved_texts = batch.get('retrieved_texts', None)
                
                outputs = model.generate(
                    pixel_values=pixel_values,
                    retrieved_texts=retrieved_texts,
                    **generation_config
                )
                
                predictions = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                references = batch.get('text', [''] * len(predictions))
                
                # Handle case where 'text' key might not exist in cached dataset
                if references is None:
                    # Fall back to reading from CSV
                    pass
                
                all_predictions.extend(predictions)
                all_references.extend(references)
                
            except Exception as e:
                print(f"\n  Error in batch {batch_idx}: {e}")
                errors += 1
                batch_size_actual = len(batch['pixel_values'])
                all_predictions.extend([""] * batch_size_actual)
                all_references.extend([""] * batch_size_actual)
            
            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
    
    print(f"  Generated {len(all_predictions)} reports ({errors} batch errors)")
    
    # --- Save raw predictions ---
    predictions_data = {
        'predictions': all_predictions,
        'references': all_references,
        'checkpoint': str(checkpoint_path),
        'val_cache': str(val_cache),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    df_preds = pd.DataFrame({
        'prediction': all_predictions,
        'reference': all_references
    })
    df_preds.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    # --- Calculate Metrics ---
    print("\n[4/4] Calculating metrics...")
    
    # Filter valid samples
    valid = [(p, r) for p, r in zip(all_predictions, all_references) 
             if p.strip() and r.strip()]
    valid_preds = [v[0] for v in valid]
    valid_refs = [v[1] for v in valid]
    
    print(f"  Valid samples: {len(valid_preds)}/{len(all_predictions)}")
    
    metrics = {}
    
    # BLEU
    print("  Computing BLEU...")
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        
        refs_tok = [[word_tokenize(r.lower())] for r in valid_refs]
        preds_tok = [word_tokenize(p.lower()) for p in valid_preds]
        
        smoothing = SmoothingFunction().method1
        
        metrics['bleu1'] = corpus_bleu(refs_tok, preds_tok, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        metrics['bleu2'] = corpus_bleu(refs_tok, preds_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        metrics['bleu3'] = corpus_bleu(refs_tok, preds_tok, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        metrics['bleu4'] = corpus_bleu(refs_tok, preds_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        
        print(f"    BLEU-1: {metrics['bleu1']:.4f}")
        print(f"    BLEU-2: {metrics['bleu2']:.4f}")
        print(f"    BLEU-3: {metrics['bleu3']:.4f}")
        print(f"    BLEU-4: {metrics['bleu4']:.4f}")
    except Exception as e:
        print(f"    BLEU failed: {e}")
    
    # ROUGE-L
    print("  Computing ROUGE-L...")
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(valid_preds, valid_refs)]
        metrics['rouge_l'] = float(np.mean(rouge_scores))
        print(f"    ROUGE-L: {metrics['rouge_l']:.4f}")
    except Exception as e:
        print(f"    ROUGE failed: {e}")
    
    # METEOR
    print("  Computing METEOR...")
    try:
        from nltk.translate.meteor_score import meteor_score
        import nltk
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
        meteor_scores = [meteor_score([word_tokenize(r.lower())], word_tokenize(p.lower())) 
                         for p, r in zip(valid_preds, valid_refs)]
        metrics['meteor'] = float(np.mean(meteor_scores))
        print(f"    METEOR: {metrics['meteor']:.4f}")
    except Exception as e:
        print(f"    METEOR failed: {e}")
    
    # Entity F1 (simple keyword matching for medical entities)
    print("  Computing Entity overlap...")
    try:
        medical_terms = [
            'effusion', 'pneumothorax', 'cardiomegaly', 'consolidation', 'edema',
            'atelectasis', 'pneumonia', 'nodule', 'mass', 'opacity', 'infiltrate',
            'pleural', 'pericardial', 'mediastinal', 'hilar', 'pulmonary',
            'catheter', 'tube', 'line', 'device', 'sternotomy', 'fracture'
        ]
        
        entity_precisions = []
        entity_recalls = []
        entity_f1s = []
        
        for pred, ref in zip(valid_preds, valid_refs):
            pred_lower = pred.lower()
            ref_lower = ref.lower()
            
            pred_entities = set(t for t in medical_terms if t in pred_lower)
            ref_entities = set(t for t in medical_terms if t in ref_lower)
            
            if len(pred_entities) == 0 and len(ref_entities) == 0:
                entity_precisions.append(1.0)
                entity_recalls.append(1.0)
                entity_f1s.append(1.0)
            elif len(pred_entities) == 0:
                entity_precisions.append(0.0)
                entity_recalls.append(0.0)
                entity_f1s.append(0.0)
            elif len(ref_entities) == 0:
                entity_precisions.append(0.0)
                entity_recalls.append(0.0)
                entity_f1s.append(0.0)
            else:
                tp = len(pred_entities & ref_entities)
                p = tp / len(pred_entities)
                r = tp / len(ref_entities)
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                entity_precisions.append(p)
                entity_recalls.append(r)
                entity_f1s.append(f1)
        
        metrics['entity_precision'] = float(np.mean(entity_precisions))
        metrics['entity_recall'] = float(np.mean(entity_recalls))
        metrics['entity_f1'] = float(np.mean(entity_f1s))
        
        print(f"    Entity F1: {metrics['entity_f1']:.4f}")
        print(f"    Entity Precision: {metrics['entity_precision']:.4f}")
        print(f"    Entity Recall: {metrics['entity_recall']:.4f}")
    except Exception as e:
        print(f"    Entity metrics failed: {e}")
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save report
    with open(os.path.join(output_dir, 'metrics_report.txt'), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION REPORT: Stage 2 Generate-Then-Retrieve\n")
        f.write("=" * 80 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Val cache:  {val_cache}\n")
        f.write(f"Samples:    {len(valid_preds)}\n")
        f.write(f"Timestamp:  {datetime.now().isoformat()}\n")
        f.write("\nMETRICS:\n")
        f.write("-" * 40 + "\n")
        for k, v in metrics.items():
            f.write(f"  {k:20s}: {v:.4f}\n")
        f.write("=" * 80 + "\n")
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 80}")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"\n  Results saved to: {output_dir}")
    print(f"{'=' * 80}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--csv_file', type=str, default='./data/processed/mimic_cxr_val_clean.csv')
    parser.add_argument('--val_cache', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results/stage2_gen_then_retrieve')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=args.checkpoint,
        csv_file=args.csv_file,
        val_cache=args.val_cache,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == '__main__':
    main()
