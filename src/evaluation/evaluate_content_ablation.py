"""
Q-Former RAG evaluation and retrieval content ablation (Section 3.3).

All models are evaluated on the same validation samples with the same decoding
(beam 5, up to 200 new tokens, no_repeat_ngram_size=3, length_penalty=1.0), then
scored at full length and truncated to 35/39/50 words (the paper's Table 3 protocol).

Models / conditions
  stage1   : Stage 1, no retrieval                                   -> s1
  qformer  : Stage 2 Q-Former RAG, with / without retrieval          -> qf_with, qf_none

Content tests (first --n_ablation samples, real images), for qformer:
  - medical (retrieved, k=3) vs pangram context (3 pangrams, one per slot):
      identical %, median word-Jaccard among non-identical (as in evaluate_expD.py),
      and median word-Jaccard over all samples
  - swapped context (image of X, retrieved reports of Y): does the output overlap
      more with Y's context than with X's own context?

Each condition's predictions are saved separately and skipped on rerun (resumable).
Models are loaded one at a time to stay within GPU memory.

Usage:
  CUDA_VISIBLE_DEVICES=0 python evaluate_content_ablation.py \
      --models stage1,qformer --output_dir ./results/content_ablation
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
import gc
import json
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from modified_qformer_opt import RAGEnhancedBLIP2_OPT

PANGRAMS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump.",
]
GEN = dict(max_new_tokens=200, num_beams=5, early_stopping=True,
           no_repeat_ngram_size=3, length_penalty=1.0)
TRUNC = [35, 39, 50]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])


# ------------------------------------------------------------------ metrics
def jaccard(a, b):
    """Word-level Jaccard, identical to word_overlap() in evaluate_expD.py."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def bleu4(preds, refs):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    for pkg, name in [('tokenizers/punkt', 'punkt'), ('tokenizers/punkt_tab', 'punkt_tab')]:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(name, quiet=True)
    pairs = [(p, r) for p, r in zip(preds, refs) if p.strip() and r.strip()]
    if not pairs:
        return 0.0, 0
    vp, vr = zip(*pairs)
    score = corpus_bleu([[word_tokenize(r.lower())] for r in vr],
                        [word_tokenize(p.lower()) for p in vp],
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=SmoothingFunction().method1)
    return float(score), len(pairs)


def truncate(text, n):
    return ' '.join(text.split()[:n])


def score_condition(preds, refs):
    out = {'avg_words': float(np.mean([len(p.split()) for p in preds if p.strip()] or [0])),
           'n_empty': int(sum(1 for p in preds if not p.strip()))}
    out['bleu4_full'], out['n_valid'] = bleu4(preds, refs)
    for n in TRUNC:
        out[f'bleu4_{n}w'], _ = bleu4([truncate(p, n) for p in preds], refs)
    return out


# ------------------------------------------------------------------ loading
def load_image(path, device):
    try:
        img = TRANSFORM(Image.open(path).convert('RGB'))
        ok = True
    except Exception:
        img, ok = torch.zeros(3, 224, 224), False
    return img.unsqueeze(0).to(device), ok


def _load_state(model, ckpt_path, prefix_check):
    try:  # memory-map to avoid holding a second 16 GB copy in RAM
        sd = torch.load(ckpt_path, map_location='cpu', mmap=True)
    except Exception:
        sd = torch.load(ckpt_path, map_location='cpu')
    sd = sd.get('model_state_dict', sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if k.startswith(prefix_check)]
    print(f"  Loaded {ckpt_path}: {len(missing)} missing, {len(unexpected)} unexpected; "
          f"missing under '{prefix_check}': {len(bad)}")
    if bad:
        print(f"  WARNING: e.g. {bad[:3]}")
    del sd


def load_rag(ckpt_path, use_retrieval, device):
    m = RAGEnhancedBLIP2_OPT(blip2_model_name='Salesforce/blip2-opt-2.7b',
                             num_query_tokens=32, freeze_vision=True,
                             freeze_llm=True, use_retrieval=use_retrieval)
    _load_state(m, ckpt_path, 'qformer.')
    return m.to(device).eval()


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ generation
@torch.no_grad()
def gen_qformer(m, pv, retrieved):
    emb = m.forward(pixel_values=pv, retrieved_texts=retrieved)
    attn = torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device)
    out = m.language_model.generate(inputs_embeds=emb, attention_mask=attn, **GEN)
    return m.tokenizer.decode(out[0], skip_special_tokens=True).strip()


# ------------------------------------------------------------------ runner
def run_condition(name, fn, indices, ctx, out_dir):
    """fn(i, pixel_values) -> text. Saves predictions; skips if already done."""
    path = os.path.join(out_dir, f'preds_{name}.json')
    if os.path.exists(path):
        print(f"  [{name}] already done, loading")
        return json.load(open(path))['predictions']
    preds, n_bad_img = [], 0
    for i in tqdm(indices, desc=name):
        pv, ok = load_image(ctx['df'].iloc[i]['image_path'], ctx['device'])
        n_bad_img += (not ok)
        try:
            preds.append(fn(i, pv))
        except Exception as e:
            print(f"  [{name}] sample {i} failed: {e}")
            preds.append("")
    json.dump({'condition': name, 'indices': list(indices), 'predictions': preds,
               'references': [ctx['refs'][i] for i in indices],
               'n_missing_images': n_bad_img, 'generation': GEN},
              open(path, 'w'), indent=2)
    return preds


def content_tests(prefix, medical, pangram, swapped, indices, ctx):
    n = len(indices)
    identical = sum(1 for a, b in zip(medical, pangram) if a.strip() == b.strip())
    diff_sims = [jaccard(a, b) for a, b in zip(medical, pangram) if a.strip() != b.strip()]
    all_sims = [jaccard(a, b) for a, b in zip(medical, pangram)]
    ctx_ov, own_ov = [], []
    for k, i in enumerate(indices):
        j = indices[(k + n // 2) % n]
        ctx_ov.append(jaccard(swapped[k], " ".join(ctx['retrieved'][j])))
        own_ov.append(jaccard(swapped[k], " ".join(ctx['retrieved'][i])))
    return {
        f'{prefix}_content_ablation': {
            'n': n,
            'identical_pct': 100.0 * identical / n,
            'median_jaccard_nonidentical': float(np.median(diff_sims)) if diff_sims else 1.0,
            'median_jaccard_all': float(np.median(all_sims)),
        },
        f'{prefix}_swapped_context': {
            'n': n,
            'mean_overlap_with_given_context': float(np.mean(ctx_ov)),
            'mean_overlap_with_own_context': float(np.mean(own_ov)),
            'given_context_wins': int(sum(c > o for c, o in zip(ctx_ov, own_ov))),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='stage1,qformer')
    ap.add_argument('--stage1_ckpt', default='./checkpoints/stage1_retrained/best_model.pt')
    ap.add_argument('--qformer_ckpt', default='./checkpoints/stage2_retrained/best_model.pt')
    ap.add_argument('--val_csv', default='./data/processed/mimic_cxr_val_clean.csv')
    ap.add_argument('--val_cache', default='./data/retrained_gtr/retrieval_cache_val_gtr.pkl')
    ap.add_argument('--n_eval', type=int, default=1000)
    ap.add_argument('--n_ablation', type=int, default=100)
    ap.add_argument('--output_dir', default='./results/content_ablation')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    df = pd.read_csv(args.val_csv)
    cache = pickle.load(open(args.val_cache, 'rb'))
    n_eval = min(args.n_eval, len(df))
    ctx = {
        'df': df, 'device': device,
        'refs': [str(x) for x in df['findings'].tolist()],
        'retrieved': {i: list(cache.get(i, [''] * 3))[:3] for i in range(n_eval)},
    }
    eval_idx = list(range(n_eval))
    abl_idx = list(range(min(args.n_ablation, n_eval)))
    n_ab = len(abl_idx)
    swap_of = {i: abl_idx[(k + n_ab // 2) % n_ab] for k, i in enumerate(abl_idx)}
    R = ctx['retrieved']

    summary_path = os.path.join(args.output_dir, 'summary.json')
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    refs_eval = [ctx['refs'][i] for i in eval_idx]
    models = [m.strip() for m in args.models.split(',') if m.strip()]

    if 'stage1' in models:
        print("\n=== Stage 1 (no retrieval) ===")
        m = load_rag(args.stage1_ckpt, use_retrieval=False, device=device)
        p = run_condition('s1', lambda i, pv: gen_qformer(m, pv, None), eval_idx, ctx, args.output_dir)
        summary['s1'] = score_condition(p, refs_eval)
        free(m)

    if 'qformer' in models:
        print("\n=== Q-Former RAG (Stage 2) ===")
        m = load_rag(args.qformer_ckpt, use_retrieval=True, device=device)
        p_with = run_condition('qf_with', lambda i, pv: gen_qformer(m, pv, [R[i]]),
                               eval_idx, ctx, args.output_dir)
        p_none = run_condition('qf_none', lambda i, pv: gen_qformer(m, pv, None),
                               eval_idx, ctx, args.output_dir)
        p_pan = run_condition('qf_pangram', lambda i, pv: gen_qformer(m, pv, [PANGRAMS]),
                              abl_idx, ctx, args.output_dir)
        p_swp = run_condition('qf_swapped', lambda i, pv: gen_qformer(m, pv, [R[swap_of[i]]]),
                              abl_idx, ctx, args.output_dir)
        summary['qf_with'] = score_condition(p_with, refs_eval)
        summary['qf_none'] = score_condition(p_none, refs_eval)
        summary.update(content_tests('qf', p_with[:n_ab], p_pan, p_swp, abl_idx, ctx))
        free(m)

    summary['_config'] = {'n_eval': n_eval, 'n_ablation': n_ab, 'generation': GEN,
                          'truncation_words': TRUNC, 'pangrams': PANGRAMS}
    json.dump(summary, open(summary_path, 'w'), indent=2)

    print("\n" + "=" * 78)
    print(f"{'condition':<12}{'words':>7}{'B4 full':>9}" + "".join(f"{'B4@'+str(n)+'w':>9}" for n in TRUNC))
    for k in ['s1', 'qf_none', 'qf_with']:
        if k in summary:
            s = summary[k]
            print(f"{k:<12}{s['avg_words']:>7.1f}{s['bleu4_full']:>9.4f}"
                  + "".join(f"{s[f'bleu4_{n}w']:>9.4f}" for n in TRUNC))
    for k in ['qf_content_ablation', 'qf_swapped_context']:
        if k in summary:
            print(k, json.dumps(summary[k]))
    print(f"Saved: {summary_path}")


if __name__ == '__main__':
    main()