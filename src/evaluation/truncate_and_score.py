"""
Truncate predictions to match GT length distribution, then recompute metrics.

This tests whether verbose outputs (from proper full-length generation) are
the reason BLEU/ROUGE dropped compared to the original truncated eval.

We truncate each prediction to match GT's length distribution - specifically,
we cap at a target word count (e.g., first 50 words matching GT mean).

Usage:
    python truncate_and_score.py \
        --predictions ./results/stage2_gen_then_retrieve_v3_lp06/predictions_with_refs.json \
        --csv ./data/processed/mimic_cxr_val_clean.csv \
        --target_length 50 \
        --output_dir ./results/stage2_gtr_truncated_50
"""

import os
import json
import argparse
import numpy as np
import pandas as pd


def truncate_to_words(text, n_words):
    """Truncate text to first n_words words, ending at sentence boundary if possible."""
    words = text.split()
    if len(words) <= n_words:
        return text
    
    truncated = ' '.join(words[:n_words])
    
    # Try to end at the last complete sentence
    for end_punct in ['. ', '! ', '? ']:
        last_idx = truncated.rfind(end_punct)
        if last_idx > len(truncated) * 0.5:  # only use if we keep at least half
            return truncated[:last_idx + 1].strip()
    
    # Otherwise just add period
    return truncated.rstrip('.,;: ') + '.'


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
        return {}
    
    vp, vr = zip(*valid)
    metrics = {'n_valid': len(valid), 'n_total': len(predictions)}
    
    refs_tok = [[word_tokenize(r.lower())] for r in vr]
    preds_tok = [word_tokenize(p.lower()) for p in vp]
    smooth = SmoothingFunction().method1
    
    metrics['bleu1'] = corpus_bleu(refs_tok, preds_tok, weights=(1, 0, 0, 0), smoothing_function=smooth)
    metrics['bleu2'] = corpus_bleu(refs_tok, preds_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    metrics['bleu3'] = corpus_bleu(refs_tok, preds_tok, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    metrics['bleu4'] = corpus_bleu(refs_tok, preds_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    metrics['rouge_l'] = float(np.mean([scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(vp, vr)]))
    
    try:
        from nltk.translate.meteor_score import meteor_score
        metrics['meteor'] = float(np.mean([meteor_score([word_tokenize(r.lower())], word_tokenize(p.lower())) for p, r in zip(vp, vr)]))
    except Exception:
        pass
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--csv', type=str, default=None,
                        help='Optional CSV to load references from if missing in predictions')
    parser.add_argument('--target_length', type=int, default=50,
                        help='Target word count for truncation (GT mean is 49)')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.predictions, 'r') as f:
        data = json.load(f)
    
    predictions = data['predictions']
    references = data.get('references', [])
    
    # Fill references from CSV if missing
    if not any(r.strip() for r in references if r) and args.csv:
        print(f"References empty, loading from CSV: {args.csv}")
        df = pd.read_csv(args.csv)
        gt_col = 'findings' if 'findings' in df.columns else 'report'
        references = df[gt_col].astype(str).tolist()
        # Align length
        n = min(len(predictions), len(references))
        predictions = predictions[:n]
        references = references[:n]
    
    print(f"\nOriginal length stats:")
    orig_lens = [len(p.split()) for p in predictions if p.strip()]
    print(f"  Mean: {np.mean(orig_lens):.0f} words, Median: {np.median(orig_lens):.0f}")
    
    print(f"\nTruncating to {args.target_length} words...")
    truncated = [truncate_to_words(p, args.target_length) for p in predictions]
    
    trunc_lens = [len(p.split()) for p in truncated if p.strip()]
    print(f"  After: Mean={np.mean(trunc_lens):.0f} words, Median={np.median(trunc_lens):.0f}")
    
    ref_lens = [len(r.split()) for r in references if r and r.strip()]
    print(f"  GT:    Mean={np.mean(ref_lens):.0f} words, Median={np.median(ref_lens):.0f}")
    
    # Compute metrics
    print(f"\nComputing metrics...")
    metrics = compute_metrics(truncated, references)
    metrics['target_length'] = args.target_length
    metrics['avg_pred_length'] = float(np.mean(trunc_lens))
    metrics['avg_ref_length'] = float(np.mean(ref_lens))
    
    # Save truncated predictions for RadGraph
    out_data = {
        'predictions': truncated,
        'references': references,
        'target_length': args.target_length,
        'source': args.predictions,
    }
    
    pred_out = os.path.join(args.output_dir, 'predictions.json')
    with open(pred_out, 'w') as f:
        json.dump(out_data, f, indent=2)
    
    metrics_out = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_out, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"METRICS (truncated to {args.target_length} words)")
    print(f"{'=' * 60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")
    print(f"{'=' * 60}")
    print(f"\nSaved:")
    print(f"  {pred_out}")
    print(f"  {metrics_out}")
    print(f"\nNext: run RadGraph F1 on the truncated predictions:")
    print(f"  python evaluate_radgraph_final.py --predictions_file {pred_out} --output_dir {args.output_dir}")


if __name__ == '__main__':
    main()
