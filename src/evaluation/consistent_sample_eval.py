"""
Consistent Sample Set Evaluation

Computes metrics for ALL conditions on the SAME first 1,000 samples
from mimic_cxr_val_clean.csv for a fair comparison table.

Conditions already on 1,000 samples (head):
  - Qwen2-VL zero-shot
  - Qwen2-VL prompt-engineered
  - BiomedCLIP image retrieval
  - ViT image retrieval
  - Oracle GT-text retrieval

Conditions that need slicing from full val set:
  - Stage 1 (no RAG) — 11,822 predictions, slice to first 1,000
  - Gen-then-retrieve — 11,822 predictions, slice to first 1,000

Usage:
    python consistent_sample_eval.py \
        --csv ./data/processed/mimic_cxr_val_clean.csv \
        --output_dir ./results/consistent_1000
"""

import os
import json
import argparse
import numpy as np
import pandas as pd


def compute_metrics(predictions, references):
    """Compute BLEU-1/4, ROUGE-L, METEOR on valid pairs"""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    from rouge_score import rouge_scorer
    import nltk

    for pkg, name in [('tokenizers/punkt', 'punkt'), ('tokenizers/punkt_tab', 'punkt_tab'),
                      ('corpora/wordnet', 'wordnet')]:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(name, quiet=True)

    valid = [(p, r) for p, r in zip(predictions, references) if p and p.strip() and r and r.strip()]
    if not valid:
        return {'n_valid': 0}

    vp, vr = zip(*valid)
    metrics = {'n_valid': len(valid)}

    refs_tok = [[word_tokenize(r.lower())] for r in vr]
    preds_tok = [word_tokenize(p.lower()) for p in vp]
    smooth = SmoothingFunction().method1

    metrics['bleu1'] = corpus_bleu(refs_tok, preds_tok, weights=(1, 0, 0, 0), smoothing_function=smooth)
    metrics['bleu4'] = corpus_bleu(refs_tok, preds_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    metrics['rouge_l'] = float(np.mean([scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(vp, vr)]))

    try:
        from nltk.translate.meteor_score import meteor_score
        metrics['meteor'] = float(np.mean([
            meteor_score([word_tokenize(r.lower())], word_tokenize(p.lower()))
            for p, r in zip(vp, vr)
        ]))
    except Exception:
        pass

    return metrics


def load_predictions(path):
    """Load predictions from JSON file"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('predictions', []), data.get('references', [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='./data/processed/mimic_cxr_val_clean.csv')
    parser.add_argument('--output_dir', type=str, default='./results/consistent_1000')
    parser.add_argument('--n_samples', type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    N = args.n_samples

    # Load CSV references (ground truth)
    df = pd.read_csv(args.csv)
    gt_col = 'findings' if 'findings' in df.columns else 'report'
    csv_refs = df[gt_col].astype(str).tolist()[:N]
    print(f"Loaded {len(csv_refs)} GT references from CSV")

    # Define all conditions and their prediction files
    conditions = {
        'Gen-Then-Retrieve': {
            'path': './results/stage2_gen_then_retrieve/predictions.json',
            'full_set': True,  # Has 11,822 predictions, needs slicing
            'refs_from_csv': True,  # References might be empty (cached dataset bug)
        },
        'Stage 1 (No RAG)': {
            'path': './results/stage1_clean/predictions.json',
            'full_set': True,
            'refs_from_csv': True,
        },
        'Image Retrieval (ViT)': {
            'path': './results/stage2_image_retrieval/predictions.json',
            'full_set': False,
            'refs_from_csv': False,
        },
        'Oracle (GT-Text)': {
            'path': './results/stage2_clean/predictions.json',
            'full_set': False,
            'refs_from_csv': False,
        },
        'BiomedCLIP Retrieval': {
            'path': './results/stage2_biomedclip_retrieval/predictions.json',
            'full_set': False,
            'refs_from_csv': False,
        },
        'Qwen2-VL (Zero-Shot)': {
            'path': './results/qwen2vl_baseline/predictions.json',
            'full_set': False,
            'refs_from_csv': False,
        },
        'Qwen2-VL (Prompt-Eng)': {
            'path': './results/qwen2vl_prompt_eng/predictions.json',
            'full_set': False,
            'refs_from_csv': False,
        },
    }

    all_results = {}

    print(f"\n{'='*80}")
    print(f"CONSISTENT {N}-SAMPLE EVALUATION")
    print(f"{'='*80}\n")

    for name, config in conditions.items():
        path = config['path']

        if not os.path.exists(path):
            print(f"  {name}: FILE NOT FOUND ({path}), skipping")
            all_results[name] = {'status': 'file_not_found'}
            continue

        preds, refs = load_predictions(path)

        # Slice to first N if full set
        if config['full_set']:
            preds = preds[:N]
            refs = refs[:N]

        # Use CSV references if predictions file has empty refs
        if config['refs_from_csv']:
            empty_refs = sum(1 for r in refs if not r or not r.strip())
            if empty_refs > len(refs) * 0.5:
                print(f"  {name}: {empty_refs}/{len(refs)} refs empty, using CSV references")
                refs = csv_refs[:len(preds)]

        # Sanity check: verify alignment
        if refs and refs[0] and csv_refs[0]:
            match = str(refs[0])[:50] == str(csv_refs[0])[:50]
            if not match and not config['refs_from_csv']:
                print(f"  {name}: WARNING - reference mismatch with CSV!")
                print(f"    Pred ref[0]: {str(refs[0])[:60]}")
                print(f"    CSV ref[0]:  {str(csv_refs[0])[:60]}")

        # Handle sample count mismatch
        actual_n = min(len(preds), len(refs), N)
        if actual_n < N:
            print(f"  {name}: Only {actual_n} samples available (need {N})")

        preds = preds[:actual_n]
        refs = refs[:actual_n]

        # Compute metrics
        print(f"  {name}: computing metrics on {actual_n} samples...")
        metrics = compute_metrics(preds, refs)
        all_results[name] = metrics

    # Print comparison table
    print(f"\n{'='*90}")
    print(f"{'Condition':35s} {'N':>5s} {'BLEU-1':>8s} {'BLEU-4':>8s} {'ROUGE-L':>8s} {'METEOR':>8s}")
    print(f"{'-'*90}")

    # Sort by BLEU-4 ascending for the gradient story
    sorted_conditions = sorted(
        all_results.items(),
        key=lambda x: x[1].get('bleu4', -1)
    )

    for name, metrics in sorted_conditions:
        if metrics.get('status') == 'file_not_found':
            print(f"  {name:33s} {'—':>5s} {'—':>8s} {'—':>8s} {'—':>8s} {'—':>8s}")
            continue

        n = metrics.get('n_valid', 0)
        b1 = metrics.get('bleu1', 0)
        b4 = metrics.get('bleu4', 0)
        rl = metrics.get('rouge_l', 0)
        mt = metrics.get('meteor', 0)
        print(f"  {name:33s} {n:5d} {b1:8.4f} {b4:8.4f} {rl:8.4f} {mt:8.4f}")

    print(f"{'='*90}")

    # Save all results
    output_file = os.path.join(args.output_dir, 'consistent_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_file}")

    # Save as markdown table for easy copy-paste into paper
    md_file = os.path.join(args.output_dir, 'comparison_table.md')
    with open(md_file, 'w') as f:
        f.write(f"# Consistent {N}-Sample Comparison\n\n")
        f.write(f"All conditions evaluated on the first {N} samples of mimic_cxr_val_clean.csv.\n\n")
        f.write(f"| Condition | N | BLEU-1 | BLEU-4 | ROUGE-L | METEOR |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        for name, metrics in sorted_conditions:
            if metrics.get('status') == 'file_not_found':
                continue
            n = metrics.get('n_valid', 0)
            b1 = metrics.get('bleu1', 0)
            b4 = metrics.get('bleu4', 0)
            rl = metrics.get('rouge_l', 0)
            mt = metrics.get('meteor', 0)
            f.write(f"| {name} | {n} | {b1:.4f} | {b4:.4f} | {rl:.4f} | {mt:.4f} |\n")
        f.write(f"\n*Note: RadGraph F1 computed separately via evaluate_radgraph_final.py*\n")

    print(f"Saved: {md_file}")

    print(f"\nNEXT STEPS:")
    print(f"  1. Review comparison_table.md")
    print(f"  2. Run RadGraph F1 on sliced gen-then-retrieve and Stage 1 predictions:")
    print(f"     (Save sliced predictions first, then run evaluate_radgraph_final.py)")


if __name__ == '__main__':
    main()
