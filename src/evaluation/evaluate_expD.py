"""
Experiment D: Post-Training Evaluation Battery

1. Swapped-context routing test (the decisive test)
   - Feed patient X's blank image with patient Y's GT report
   - If output tracks Y → retrieval is routing
   - If output is generic → model is cheating

2. Content ablation rerun (medical vs garbage)
   - Same 100-sample protocol
   - Prediction: if pathway trained, 92.5% similarity should collapse

3. Standard generation for BLEU/RadGraph
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
import torch
import json
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

from modified_qformer_opt import RAGEnhancedBLIP2_OPT


def load_model(checkpoint_path, device):
    model = RAGEnhancedBLIP2_OPT(
        blip2_model_name='Salesforce/blip2-opt-2.7b',
        num_query_tokens=32,
        freeze_vision=True,
        freeze_llm=True,
        use_retrieval=True
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model


def create_blank_image():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    blank = Image.new('RGB', (224, 224), color=(0, 0, 0))
    return transform(blank)


def word_overlap(text_a, text_b):
    """Compute word-level Jaccard similarity."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


# ============================================================
# TEST 1: Swapped-Context Routing Test
# ============================================================

def swapped_context_test(model, csv_file, device, n_samples=100):
    """
    Feed patient X's blank image with patient Y's GT report.
    If output tracks Y's report → retrieval routes information.
    If output is generic → model found a shortcut.
    """
    print("\n" + "=" * 60)
    print("TEST 1: SWAPPED-CONTEXT ROUTING")
    print("=" * 60)
    
    df = pd.read_csv(csv_file)
    blank_image = create_blank_image().unsqueeze(0).to(device)
    
    # Pick n_samples pairs
    indices = list(range(min(n_samples * 2, len(df))))
    np.random.shuffle(indices)
    
    context_tracking = []
    own_tracking = []
    
    for i in range(0, min(n_samples * 2, len(indices)), 2):
        idx_x = indices[i]
        idx_y = indices[i + 1] if i + 1 < len(indices) else indices[0]
        
        report_x = str(df.iloc[idx_x]['findings'])
        report_y = str(df.iloc[idx_y]['findings'])
        
        # Generate with Y's context (swapped)
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values=blank_image,
                retrieved_texts=[[report_y]],
                max_length=100,
                num_beams=5,
                length_penalty=1.0
            )
            generated = model.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
        
        # Does output track Y's report (the context)?
        overlap_with_context = word_overlap(generated, report_y)
        overlap_with_own = word_overlap(generated, report_x)
        
        context_tracking.append(overlap_with_context)
        own_tracking.append(overlap_with_own)
    
    results = {
        'mean_context_overlap': float(np.mean(context_tracking)),
        'mean_own_overlap': float(np.mean(own_tracking)),
        'median_context_overlap': float(np.median(context_tracking)),
        'n_samples': len(context_tracking),
        'context_wins': int(sum(1 for c, o in zip(context_tracking, own_tracking) if c > o)),
    }
    
    print(f"  Samples tested: {results['n_samples']}")
    print(f"  Mean overlap with CONTEXT (Y): {results['mean_context_overlap']:.4f}")
    print(f"  Mean overlap with OWN (X):     {results['mean_own_overlap']:.4f}")
    print(f"  Context > Own: {results['context_wins']}/{results['n_samples']}")
    print()
    
    if results['mean_context_overlap'] > results['mean_own_overlap'] + 0.05:
        print("  VERDICT: Retrieval IS routing information ✅")
        print("  Output tracks the context report, not the image's own report.")
    elif abs(results['mean_context_overlap'] - results['mean_own_overlap']) < 0.05:
        print("  VERDICT: Output is GENERIC (shortcut) ⚠️")
        print("  Output is similar to both reports — likely templated.")
    else:
        print("  VERDICT: Output tracks OWN report ❌")
        print("  Model is using residual visual info, not retrieval.")
    
    return results


# ============================================================
# TEST 2: Content Ablation (medical vs garbage)
# ============================================================

def content_ablation_test(model, csv_file, device, n_samples=100):
    """
    Compare outputs with medical vs irrelevant retrieval context.
    If pathway trained: outputs should differ significantly.
    If pathway still starved: 92.5% similarity (same as before).
    """
    print("\n" + "=" * 60)
    print("TEST 2: CONTENT ABLATION (MEDICAL vs GARBAGE)")
    print("=" * 60)
    
    df = pd.read_csv(csv_file)
    blank_image = create_blank_image().unsqueeze(0).to(device)
    
    garbage_texts = [
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump."
    ]
    
    identical = 0
    word_similarities = []
    
    for i in tqdm(range(min(n_samples, len(df))), desc="Content ablation"):
        report = str(df.iloc[i]['findings'])
        
        with torch.no_grad():
            # Medical context
            out_medical = model.generate(
                pixel_values=blank_image,
                retrieved_texts=[[report]],
                max_length=100, num_beams=5, length_penalty=1.0
            )
            gen_medical = model.tokenizer.batch_decode(
                out_medical, skip_special_tokens=True
            )[0]
            
            # Garbage context
            out_garbage = model.generate(
                pixel_values=blank_image,
                retrieved_texts=[garbage_texts],
                max_length=100, num_beams=5, length_penalty=1.0
            )
            gen_garbage = model.tokenizer.batch_decode(
                out_garbage, skip_special_tokens=True
            )[0]
        
        if gen_medical.strip() == gen_garbage.strip():
            identical += 1
        else:
            sim = word_overlap(gen_medical, gen_garbage)
            word_similarities.append(sim)
    
    results = {
        'n_samples': min(n_samples, len(df)),
        'identical_pct': identical / min(n_samples, len(df)) * 100,
        'median_similarity': float(np.median(word_similarities)) if word_similarities else 1.0,
        'mean_similarity': float(np.mean(word_similarities)) if word_similarities else 1.0,
    }
    
    print(f"  Identical outputs: {identical}/{results['n_samples']} ({results['identical_pct']:.1f}%)")
    print(f"  Among different: median similarity = {results['median_similarity']:.4f}")
    print()
    
    if results['identical_pct'] < 20 and results['median_similarity'] < 0.7:
        print("  VERDICT: Model USES retrieval content ✅")
        print("  Outputs differ significantly between medical and garbage.")
    else:
        print("  VERDICT: Model IGNORES retrieval content ❌")
        print(f"  Previous baseline: 42% identical, 92.5% similarity")
        print(f"  Current: {results['identical_pct']:.0f}% identical, "
              f"{results['median_similarity']:.1%} similarity")
    
    return results


# ============================================================
# TEST 3: Weight Analysis
# ============================================================

def weight_analysis(model, stage1_checkpoint):
    """Compare final weights to Stage 1 initialization."""
    print("\n" + "=" * 60)
    print("TEST 3: WEIGHT ANALYSIS")
    print("=" * 60)
    
    # Load Stage 1 weights for comparison
    s1_ckpt = torch.load(stage1_checkpoint, map_location='cpu')
    s1_state = s1_ckpt.get('model_state_dict', s1_ckpt)
    
    results = {}
    for name, param in model.named_parameters():
        if 'retrieval_cross_attention' in name and 'weight' in name:
            final_std = param.data.std().item()
            
            if name in s1_state:
                init_val = s1_state[name]
                change = (param.data.cpu() - init_val).abs().mean().item()
            else:
                change = 0.0
            
            results[name] = {
                'final_std': final_std,
                'mean_abs_change': change
            }
            print(f"  {name.split('.')[-2]}.{name.split('.')[-1]}:")
            print(f"    Final std: {final_std:.6f}")
            print(f"    Mean abs change from init: {change:.6f}")
    
    # Compare with visual cross-attention benchmark (0.02-0.04)
    print(f"\n  Visual CA benchmark: mean abs change 0.02-0.04")
    retrieval_changes = [v['mean_abs_change'] for v in results.values()]
    avg_change = np.mean(retrieval_changes) if retrieval_changes else 0
    print(f"  Retrieval CA average change: {avg_change:.6f}")
    
    if avg_change > 0.01:
        print("  VERDICT: Retrieval pathway TRAINED ✅")
    else:
        print("  VERDICT: Retrieval pathway UNCHANGED ❌")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--stage1_checkpoint', 
        default='./checkpoints/stage1_retrained/best_model.pt')
    parser.add_argument('--csv_file',
        default='./data/processed/mimic_cxr_val_clean.csv')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device)
    
    # Run all tests
    results = {}
    
    results['swapped_context'] = swapped_context_test(
        model, args.csv_file, device, args.n_samples
    )
    
    results['content_ablation'] = content_ablation_test(
        model, args.csv_file, device, args.n_samples
    )
    
    results['weight_analysis'] = weight_analysis(
        model, args.stage1_checkpoint
    )
    
    # Save
    with open(os.path.join(args.output_dir, 'expD_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("ALL TESTS COMPLETE")
    print(f"Results saved to {args.output_dir}/expD_results.json")
    print(f"{'=' * 60}")
