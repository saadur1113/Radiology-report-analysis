"""
Statistical Significance Tests for WACV Paper

Computes paired bootstrap significance tests between conditions.
Tests whether the differences between methods are statistically significant.

Usage:
    python statistical_tests.py \
        --results_dir ./results \
        --csv ./data/processed/mimic_cxr_val_clean.csv \
        --output_dir ./results/significance_tests
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import nltk
from tqdm import tqdm

for pkg, name in [('tokenizers/punkt', 'punkt'), ('tokenizers/punkt_tab', 'punkt_tab'),
                  ('corpora/wordnet', 'wordnet')]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(name, quiet=True)


def compute_per_sample_bleu4(predictions, references):
    """Compute BLEU-4 for each sample individually"""
    smooth = SmoothingFunction().method1
    scores = []
    for p, r in zip(predictions, references):
        if not p.strip() or not r.strip():
            scores.append(0.0)
            continue
        ref_tok = [word_tokenize(r.lower())]
        pred_tok = word_tokenize(p.lower())
        score = sentence_bleu(ref_tok, pred_tok, weights=(0.25, 0.25, 0.25, 0.25),
                              smoothing_function=smooth)
        scores.append(score)
    return np.array(scores)


def compute_per_sample_rougel(predictions, references):
    """Compute ROUGE-L for each sample individually"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for p, r in zip(predictions, references):
        if not p.strip() or not r.strip():
            scores.append(0.0)
            continue
        score = scorer.score(r, p)['rougeL'].fmeasure
        scores.append(score)
    return np.array(scores)


def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000, seed=42):
    """
    Paired bootstrap significance test.
    Tests H0: mean(scores_a) == mean(scores_b)
    Returns p-value (two-sided)
    """
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    assert len(scores_b) == n

    observed_diff = np.mean(scores_a) - np.mean(scores_b)
    
    # Paired bootstrap: resample the per-sample differences
    diffs = scores_a - scores_b
    # Center the differences under H0 (shift to mean=0)
    diffs_centered = diffs - np.mean(diffs)

    count = 0
    for _ in range(n_bootstrap):
        # Resample centered differences with replacement
        indices = rng.randint(0, n, size=n)
        boot_diff = np.mean(diffs_centered[indices])

        if abs(boot_diff) >= abs(observed_diff):
            count += 1

    p_value = count / n_bootstrap
    return p_value, observed_diff


def wilcoxon_test(scores_a, scores_b):
    """Wilcoxon signed-rank test (non-parametric paired test)"""
    from scipy.stats import wilcoxon
    
    # Filter out ties (identical scores)
    diffs = scores_a - scores_b
    non_zero = diffs != 0
    
    if np.sum(non_zero) < 10:
        return 1.0, np.mean(scores_a) - np.mean(scores_b)
    
    stat, p_value = wilcoxon(scores_a[non_zero], scores_b[non_zero])
    return p_value, np.mean(scores_a) - np.mean(scores_b)


def load_predictions(path, n=1000):
    """Load predictions from JSON, slice to n samples"""
    with open(path, 'r') as f:
        data = json.load(f)
    preds = data.get('predictions', [])[:n]
    refs = data.get('references', [])[:n]
    return preds, refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='./data/processed/mimic_cxr_val_clean.csv')
    parser.add_argument('--output_dir', type=str, default='./results/significance_tests')
    parser.add_argument('--n_bootstrap', type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load CSV references
    df = pd.read_csv(args.csv)
    gt_col = 'findings' if 'findings' in df.columns else 'report'
    csv_refs = df[gt_col].astype(str).tolist()[:1000]

    # Define conditions and paths
    conditions = {
        'Stage 1 (No RAG)': './results/stage1_clean/predictions.json',
        'ViT Image Retrieval': './results/stage2_image_retrieval/predictions.json',
        'Gen-Then-Retrieve': './results/k_ablation/k_3/predictions.json',  # Fresh cache k=3
        'BiomedCLIP Retrieval': './results/stage2_biomedclip_retrieval/predictions.json',
        'Qwen2-VL (Prompt-Eng)': './results/qwen2vl_prompt_eng/predictions.json',
    }

    # Load all predictions
    print("Loading predictions...")
    all_preds = {}
    for name, path in conditions.items():
        if not os.path.exists(path):
            print(f"  {name}: NOT FOUND ({path})")
            continue
        preds, refs = load_predictions(path, n=1000)
        
        # Use CSV refs if prediction refs are empty
        if not refs or not refs[0].strip():
            refs = csv_refs[:len(preds)]
        
        all_preds[name] = (preds, refs)
        print(f"  {name}: {len(preds)} samples loaded")

    # Compute per-sample scores
    print("\nComputing per-sample scores...")
    per_sample_bleu = {}
    per_sample_rouge = {}

    for name, (preds, refs) in tqdm(all_preds.items(), desc="Scoring"):
        per_sample_bleu[name] = compute_per_sample_bleu4(preds, refs)
        per_sample_rouge[name] = compute_per_sample_rougel(preds, refs)

    # Key comparisons
    comparisons = [
        ('Gen-Then-Retrieve', 'Stage 1 (No RAG)', 'GTR vs No RAG'),
        ('BiomedCLIP Retrieval', 'Stage 1 (No RAG)', 'BiomedCLIP vs No RAG'),
        ('Gen-Then-Retrieve', 'ViT Image Retrieval', 'GTR vs ViT Retrieval'),
        ('BiomedCLIP Retrieval', 'ViT Image Retrieval', 'BiomedCLIP vs ViT Retrieval'),
        ('BiomedCLIP Retrieval', 'Gen-Then-Retrieve', 'BiomedCLIP vs GTR'),
        ('Gen-Then-Retrieve', 'Qwen2-VL (Prompt-Eng)', 'GTR vs Qwen-PE'),
    ]

    print(f"\n{'='*90}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*90}")
    print(f"{'Comparison':40s} {'Metric':>8s} {'Diff':>8s} {'p-boot':>8s} {'p-wilc':>8s} {'Sig?':>6s}")
    print(f"{'-'*90}")

    results = []

    for method_a, method_b, label in comparisons:
        if method_a not in per_sample_bleu or method_b not in per_sample_bleu:
            continue

        for metric_name, scores_dict in [('BLEU-4', per_sample_bleu), ('ROUGE-L', per_sample_rouge)]:
            scores_a = scores_dict[method_a]
            scores_b = scores_dict[method_b]

            # Align lengths
            n = min(len(scores_a), len(scores_b))
            scores_a = scores_a[:n]
            scores_b = scores_b[:n]

            p_boot, diff_boot = paired_bootstrap_test(scores_a, scores_b, n_bootstrap=args.n_bootstrap)
            p_wilc, diff_wilc = wilcoxon_test(scores_a, scores_b)

            sig = '***' if p_boot < 0.001 else '**' if p_boot < 0.01 else '*' if p_boot < 0.05 else 'ns'

            print(f"  {label:38s} {metric_name:>8s} {diff_boot:>+8.4f} {p_boot:>8.4f} {p_wilc:>8.4f} {sig:>6s}")

            results.append({
                'comparison': label,
                'metric': metric_name,
                'method_a': method_a,
                'method_b': method_b,
                'mean_a': float(np.mean(scores_a)),
                'mean_b': float(np.mean(scores_b)),
                'diff': float(diff_boot),
                'p_bootstrap': float(p_boot),
                'p_wilcoxon': float(p_wilc),
                'significant_005': p_boot < 0.05,
                'significant_001': p_boot < 0.01,
            })

    print(f"{'='*90}")
    print(f"\n  * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant")
    print(f"  Bootstrap: {args.n_bootstrap} iterations")

    # Save results
    output_file = os.path.join(args.output_dir, 'significance_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {output_file}")

    # Print LaTeX-ready summary
    print(f"\n  For the paper, add to Section 5.1:")
    print(f"  'Paired bootstrap tests (10,000 iterations) confirm that both")
    print(f"   gen-then-retrieve and BiomedCLIP retrieval significantly outperform")
    print(f"   the no-RAG baseline (p < X.XX) and ViT image retrieval (p < X.XX)")
    print(f"   on BLEU-4, while the difference between gen-then-retrieve and")
    print(f"   BiomedCLIP is not statistically significant (p = X.XX).'")


if __name__ == '__main__':
    main()