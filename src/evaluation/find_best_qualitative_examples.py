"""
Find Best Qualitative Examples for WACV Paper Figure

Systematically searches predictions to find clear examples of:
  (a) RAG helps — BiomedCLIP captures findings Stage 1 misses
  (b) RAG neutral — both produce similar quality  
  (c) RAG hallucinates — BiomedCLIP introduces wrong findings
  (d) Qwen2-VL templated — produces "normal" for pathological cases

Outputs the text for each case + identifies the CXR image paths.

Usage:
    python find_best_qualitative_examples.py
"""

import json
import os
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from collections import Counter

# ============================================================
# PATHS — adjust if needed
# ============================================================
BASE = '.'

STAGE1_PREDS = os.path.join(BASE, 'results/retrained/stage1/predictions.json')
BIO_PREDS = os.path.join(BASE, 'results/retrained/biomedclip/predictions.json')
VIT_PREDS = os.path.join(BASE, 'results/retrained/vit_retrieval/predictions.json')
VAL_CSV = os.path.join(BASE, 'data/processed/mimic_cxr_val_clean.csv')

# Qwen2-VL predictions (if available)
QWEN_PREDS = os.path.join(BASE, 'results/qwen2vl_baseline/predictions.json')
QWEN_PE_PREDS = os.path.join(BASE, 'results/qwen2vl_baseline/predictions_prompt_eng.json')

# ============================================================
# LOAD DATA
# ============================================================
print("Loading predictions...")

def load_preds(path):
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get('predictions', d)

s1_preds = load_preds(STAGE1_PREDS)
bio_preds = load_preds(BIO_PREDS)
vit_preds = load_preds(VIT_PREDS)
qwen_preds = load_preds(QWEN_PE_PREDS) or load_preds(QWEN_PREDS)

# Load references from CSV
df = pd.read_csv(VAL_CSV)
gt_col = 'findings' if 'findings' in df.columns else 'report'
references = df[gt_col].astype(str).tolist()
image_paths = df['image_path'].tolist() if 'image_path' in df.columns else [None] * len(references)

n_samples = min(len(s1_preds or []), len(bio_preds or []), len(references))
if qwen_preds:
    n_samples = min(n_samples, len(qwen_preds))
print(f"  Comparing {n_samples} samples")

# ============================================================
# MEDICAL KEYWORDS for pathology detection
# ============================================================
PATHOLOGY_KEYWORDS = [
    'effusion', 'pneumothorax', 'consolidation', 'atelectasis',
    'edema', 'cardiomegaly', 'enlarged', 'opacity', 'opacities',
    'infiltrate', 'mass', 'nodule', 'fracture', 'pleural',
    'pneumonia', 'congestion', 'thickening', 'widening',
    'sternotomy', 'cabg', 'catheter', 'tube', 'line', 'pacer',
    'pacemaker', 'stent', 'hardware', 'prosthesis', 'wire',
    'picc', 'port', 'drain', 'tracheostomy', 'device'
]

DEVICE_KEYWORDS = [
    'catheter', 'tube', 'line', 'pacer', 'pacemaker', 'stent',
    'hardware', 'wire', 'picc', 'port', 'drain', 'tracheostomy',
    'device', 'sternotomy', 'cabg', 'prosthesis', 'valve'
]

NEGATION_PATTERNS = [
    'no ', 'without ', 'negative ', 'absent ', 'unremarkable',
    'normal', 'clear', 'within normal limits', 'no evidence',
    'no acute', 'no focal'
]

def extract_findings(text):
    """Extract mentioned pathologies/devices from text."""
    text_lower = text.lower()
    found = set()
    for kw in PATHOLOGY_KEYWORDS:
        if kw in text_lower:
            # Check if negated
            negated = False
            idx = text_lower.find(kw)
            context = text_lower[max(0, idx-25):idx]
            for neg in NEGATION_PATTERNS:
                if neg in context:
                    negated = True
                    break
            found.add((kw, negated))
    return found

def has_pathology(text):
    """Check if GT report describes pathological findings."""
    findings = extract_findings(text)
    positive = [kw for kw, neg in findings if not neg]
    return len(positive) >= 2  # at least 2 positive findings

def similarity(a, b):
    """Word-level similarity ratio."""
    return SequenceMatcher(None, a.lower().split(), b.lower().split()).ratio()

def truncate(text, max_words=50):
    """Truncate to max words for display."""
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '...'
    return text

# ============================================================
# ANALYZE EACH SAMPLE
# ============================================================
print("\nAnalyzing samples...")

results = {
    'rag_helps': [],      # BiomedCLIP captures findings Stage 1 misses
    'rag_neutral': [],    # Both produce similar quality
    'rag_hallucinates': [],  # BiomedCLIP introduces wrong findings
    'qwen_templated': [],    # Qwen produces templated output for pathological case
}

for i in range(n_samples):
    gt = references[i]
    s1 = s1_preds[i] if s1_preds else ""
    bio = bio_preds[i] if bio_preds else ""
    qwen = qwen_preds[i] if qwen_preds and i < len(qwen_preds) else ""
    img = image_paths[i] if i < len(image_paths) else ""
    
    if not gt.strip() or not s1.strip() or not bio.strip():
        continue
    
    # Extract findings
    gt_findings = extract_findings(gt)
    s1_findings = extract_findings(s1)
    bio_findings = extract_findings(bio)
    
    gt_positive = {kw for kw, neg in gt_findings if not neg}
    s1_positive = {kw for kw, neg in s1_findings if not neg}
    bio_positive = {kw for kw, neg in bio_findings if not neg}
    
    # Things GT mentions positively
    gt_negated = {kw for kw, neg in gt_findings if neg}
    
    # What each method captures correctly
    s1_correct = gt_positive & s1_positive
    bio_correct = gt_positive & bio_positive
    
    # What each method misses (GT positive, method doesn't mention or negates)
    s1_missed = gt_positive - s1_positive
    bio_missed = gt_positive - bio_positive
    
    # What each method hallucinates (method positive, GT negates or doesn't mention)
    s1_halluc = s1_positive - gt_positive
    bio_halluc = bio_positive - gt_positive
    
    # Similarity between Stage 1 and BiomedCLIP
    sim = similarity(s1, bio)
    
    # Is this a pathological case?
    is_pathological = has_pathology(gt)
    
    info = {
        'index': i,
        'gt': gt,
        's1': s1,
        'bio': bio,
        'qwen': qwen,
        'image_path': img,
        'gt_positive': gt_positive,
        's1_correct': s1_correct,
        'bio_correct': bio_correct,
        's1_missed': s1_missed,
        'bio_missed': bio_missed,
        's1_halluc': s1_halluc,
        'bio_halluc': bio_halluc,
        'similarity': sim,
        'is_pathological': is_pathological,
    }
    
    # ---- Category: RAG HELPS ----
    # BiomedCLIP captures GT findings that Stage 1 misses
    bio_advantage = bio_correct - s1_correct
    if len(bio_advantage) >= 2 and len(bio_halluc) <= 1 and is_pathological:
        info['bio_advantage'] = bio_advantage
        info['score'] = len(bio_advantage) - len(bio_halluc)
        results['rag_helps'].append(info)
    
    # ---- Category: RAG NEUTRAL ----
    # Both produce similar reports (high similarity, similar findings)
    if sim > 0.7 and abs(len(s1_correct) - len(bio_correct)) <= 1 and is_pathological:
        info['score'] = sim
        results['rag_neutral'].append(info)
    
    # ---- Category: RAG HALLUCINATES ----
    # BiomedCLIP introduces findings NOT in GT
    if len(bio_halluc) >= 2 and len(s1_halluc) <= 1 and is_pathological:
        info['score'] = len(bio_halluc) - len(s1_halluc)
        results['rag_hallucinates'].append(info)
    
    # ---- Category: QWEN TEMPLATED ----
    # Qwen produces normal/templated output for pathological case
    if qwen and is_pathological:
        qwen_lower = qwen.lower()
        is_templated = (
            'normal' in qwen_lower or
            'no acute' in qwen_lower or
            'unremarkable' in qwen_lower
        )
        qwen_positive = {kw for kw, neg in extract_findings(qwen) if not neg}
        qwen_misses = gt_positive - qwen_positive
        if is_templated and len(qwen_misses) >= 2:
            info['qwen_misses'] = qwen_misses
            info['score'] = len(qwen_misses)
            results['qwen_templated'].append(info)

# ============================================================
# RANK AND DISPLAY TOP EXAMPLES
# ============================================================
print("\n" + "=" * 70)
print("TOP QUALITATIVE EXAMPLES")
print("=" * 70)

for category, examples in results.items():
    examples.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    print(f"\n{'=' * 60}")
    print(f"  CATEGORY: {category.upper()} ({len(examples)} candidates)")
    print(f"{'=' * 60}")
    
    for rank, ex in enumerate(examples[:5]):  # Top 5 per category
        print(f"\n  --- Rank {rank+1} (index={ex['index']}, score={ex.get('score', 0):.2f}) ---")
        print(f"  Image: {ex['image_path']}")
        print(f"  GT positive findings: {ex['gt_positive']}")
        print(f"  S1 correct: {ex['s1_correct']}, missed: {ex['s1_missed']}")
        print(f"  Bio correct: {ex['bio_correct']}, missed: {ex['bio_missed']}")
        print(f"  S1 hallucinated: {ex['s1_halluc']}")
        print(f"  Bio hallucinated: {ex['bio_halluc']}")
        if 'bio_advantage' in ex:
            print(f"  Bio advantage: {ex['bio_advantage']}")
        print(f"  Similarity: {ex['similarity']:.3f}")
        print(f"\n  GT:  {truncate(ex['gt'], 50)}")
        print(f"  S1:  {truncate(ex['s1'], 50)}")
        print(f"  Bio: {truncate(ex['bio'], 50)}")
        if ex.get('qwen'):
            print(f"  Qwen: {truncate(ex['qwen'], 50)}")
        print()

# ============================================================
# SAVE BEST EXAMPLES FOR PAPER
# ============================================================
print("\n" + "=" * 70)
print("RECOMMENDED FIGURE SELECTIONS")
print("=" * 70)

selections = {}
for category in ['rag_helps', 'rag_neutral', 'rag_hallucinates', 'qwen_templated']:
    if results[category]:
        best = results[category][0]
        selections[category] = best
        print(f"\n{category}: index={best['index']}")
        print(f"  Image: {best['image_path']}")

# Save selections to JSON for reference
output_path = os.path.join(BASE, 'results/qualitative_selections.json')
saved = {}
for cat, ex in selections.items():
    saved[cat] = {
        'index': ex['index'],
        'image_path': ex['image_path'],
        'gt': ex['gt'],
        's1': ex['s1'],
        'bio': ex['bio'],
        'qwen': ex.get('qwen', ''),
        'gt_positive': list(ex['gt_positive']),
        's1_correct': list(ex['s1_correct']),
        'bio_correct': list(ex['bio_correct']),
        's1_halluc': list(ex['s1_halluc']),
        'bio_halluc': list(ex['bio_halluc']),
    }

with open(output_path, 'w') as f:
    json.dump(saved, f, indent=2)
print(f"\nSaved selections: {output_path}")

# ============================================================
# GENERATE LATEX FOR SELECTED EXAMPLES
# ============================================================
print("\n" + "=" * 70)
print("LATEX SNIPPETS FOR EACH SELECTED EXAMPLE")
print("=" * 70)

for cat, ex in selections.items():
    print(f"\n% --- {cat.upper()} (index={ex['index']}) ---")
    print(f"% Image: {ex['image_path']}")
    print(f"\\textbf{{GT:}} {ex['gt'][:200]}")
    print(f"\\textbf{{Stage 1:}} {ex['s1'][:200]}")
    print(f"\\textbf{{BiomedCLIP:}} {ex['bio'][:200]}")
    if ex.get('qwen'):
        print(f"\\textbf{{Qwen2-VL:}} {ex['qwen'][:200]}")

print(f"\n{'=' * 70}")
print("DONE")
print(f"{'=' * 70}")
print(f"\nNext steps:")
print(f"1. Review the top candidates above")
print(f"2. Pick the clearest example from each category")
print(f"3. Copy the selected CXR images into the figure directory")
print(f"4. Update qualitative_figure_final.tex with new text")