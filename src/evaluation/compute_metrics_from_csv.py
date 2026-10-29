"""
Compute metrics for evaluation results that are missing references.

The cached dataset doesn't expose references, so predictions.json had
empty 'references' entries. This script reads references from the CSV
(in CSV order, since the eval ran in CSV order) and computes all metrics.

Usage:
    python compute_metrics_from_csv.py \
        --predictions ./results/stage2_gen_then_retrieve_v2/predictions.json \
        --csv ./data/processed/mimic_cxr_val_clean.csv \
        --output_dir ./results/stage2_gen_then_retrieve_v2
"""

import os
import json
import argparse
import numpy as np
import pandas as pd


def compute_metrics(predictions, references):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    from rouge_score import rouge_scorer
    import nltk
    
    for pkg, name in [('tokenizers/punkt', 'punkt'), ('tokenizers/punkt_tab', 'punkt_tab'), ('corpora/wordnet', 'wordnet')]:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(name, quiet=True)
    
    valid = [(p, r) for p, r in zip(predictions, references) if p and p.strip() and r and r.strip()]
    if not valid:
        print("ERROR: No valid (pred, ref) pairs!")
        return {}
    
    vp = [v[0] for v in valid]
    vr = [v[1] for v in valid]
    
    metrics = {'n_valid': len(valid), 'n_total': len(predictions)}
    
    print(f"  Valid pairs: {len(valid)}/{len(predictions)}")
    
    print("  Computing BLEU...")
    refs_tok = [[word_tokenize(r.lower())] for r in vr]
    preds_tok = [word_tokenize(p.lower()) for p in vp]
    smooth = SmoothingFunction().method1
    
    metrics['bleu1'] = corpus_bleu(refs_tok, preds_tok, weights=(1, 0, 0, 0), smoothing_function=smooth)
    metrics['bleu2'] = corpus_bleu(refs_tok, preds_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    metrics['bleu3'] = corpus_bleu(refs_tok, preds_tok, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    metrics['bleu4'] = corpus_bleu(refs_tok, preds_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    print("  Computing ROUGE-L...")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    metrics['rouge_l'] = float(np.mean([scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(vp, vr)]))
    
    print("  Computing METEOR...")
    try:
        from nltk.translate.meteor_score import meteor_score
        metrics['meteor'] = float(np.mean([meteor_score([word_tokenize(r.lower())], word_tokenize(p.lower())) for p, r in zip(vp, vr)]))
    except Exception as e:
        print(f"    METEOR failed: {e}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    print(f"Loading predictions: {args.predictions}")
    with open(args.predictions, 'r') as f:
        data = json.load(f)
    
    predictions = data['predictions']
    print(f"  N predictions: {len(predictions)}")
    
    print(f"Loading CSV references: {args.csv}")
    df = pd.read_csv(args.csv)
    gt_col = 'findings' if 'findings' in df.columns else 'report'
    references = df[gt_col].astype(str).tolist()
    print(f"  N CSV rows: {len(references)}")
    
    if len(predictions) != len(references):
        print(f"  WARNING: count mismatch ({len(predictions)} preds vs {len(references)} refs)")
        print(f"  Truncating to min({len(predictions)}, {len(references)})")
        n = min(len(predictions), len(references))
        predictions = predictions[:n]
        references = references[:n]
    
    # Sanity check: are predictions and references plausibly aligned?
    print(f"\nSanity check first sample:")
    print(f"  Pred: {predictions[0][:150]}")
    print(f"  Ref:  {references[0][:150]}")
    
    print(f"\nComputing metrics...")
    metrics = compute_metrics(predictions, references)
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Update predictions.json with references for downstream tools (RadGraph, qual examples)
    data['references'] = references[:len(data['predictions'])]
    if len(references) < len(data['predictions']):
        data['references'].extend([''] * (len(data['predictions']) - len(references)))
    
    pred_file_with_refs = args.predictions.replace('.json', '_with_refs.json')
    with open(pred_file_with_refs, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved updated predictions with references: {pred_file_with_refs}")
    
    print(f"\n{'=' * 60}")
    print(f"METRICS")
    print(f"{'=' * 60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:15s}: {v:.4f}")
        else:
            print(f"  {k:15s}: {v}")
    print(f"{'=' * 60}")
    print(f"\nSaved: {metrics_file}")


if __name__ == '__main__':
    main()
