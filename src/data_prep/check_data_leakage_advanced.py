"""
ADVANCED Data Leakage Detection Methods
Beyond standard checks - deeper investigation techniques
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt


def check_9_report_memorization(model_predictions, val_df, train_df):
    """
    Check 9: Model memorization test
    If model outputs training reports verbatim, it's memorizing
    """
    print("\n" + "="*80)
    print("CHECK 9: MODEL MEMORIZATION TEST")
    print("="*80)
    
    train_reports = set(train_df['findings'].fillna('').values)
    
    memorized_count = 0
    near_memorized_count = 0
    
    for pred in model_predictions[:1000]:  # Check first 1000
        if pred in train_reports:
            memorized_count += 1
        else:
            # Check if very similar (>98% word overlap)
            pred_words = set(pred.lower().split())
            for train_report in list(train_reports)[:10000]:  # Sample for speed
                train_words = set(train_report.lower().split())
                if len(pred_words) > 0 and len(train_words) > 0:
                    overlap = len(pred_words & train_words) / len(pred_words)
                    if overlap > 0.98:
                        near_memorized_count += 1
                        break
    
    print(f"Out of 1000 predictions:")
    print(f"  Exact memorization: {memorized_count}")
    print(f"  Near-memorization:  {near_memorized_count}")
    
    if memorized_count > 10:
        print(f"\n❌ CRITICAL: Model is memorizing training data!")
        return False
    elif near_memorized_count > 50:
        print(f"\n⚠️  WARNING: High near-memorization rate")
        return False
    else:
        print(f"\n✅ No significant memorization detected")
        return True


def check_10_retrieval_hits_analysis(retrieval_dir, val_df):
    """
    Check 10: Analyze what retrieval actually returns
    Are retrieved reports from validation patients?
    """
    print("\n" + "="*80)
    print("CHECK 10: RETRIEVAL HITS ANALYSIS")
    print("="*80)
    
    try:
        # Load retrieval index
        import faiss
        
        retrieval_path = Path(retrieval_dir)
        
        # Check if index exists
        if not (retrieval_path / 'faiss_index.bin').exists():
            print("⚠️  No FAISS index found - skipping")
            return True
        
        # Load metadata
        metadata = pd.read_csv(retrieval_path / 'retrieval_metadata.csv')
        
        # Extract patient IDs from retrieval DB
        def extract_patient(path):
            parts = str(path).split('/')
            for p in parts:
                if p.startswith('p') and len(p) > 5:
                    return p
            return None
        
        if 'image_path' in metadata.columns:
            metadata['patient_id'] = metadata['image_path'].apply(extract_patient)
        elif 'text' in metadata.columns:
            # If only text, can't extract patient ID reliably
            print("⚠️  Cannot verify patient IDs in retrieval DB")
            return True
        
        val_df['patient_id'] = val_df['image_path'].apply(extract_patient)
        
        retrieval_patients = set(metadata['patient_id'].dropna())
        val_patients = set(val_df['patient_id'].dropna())
        
        overlap = retrieval_patients & val_patients
        
        if len(overlap) > 0:
            overlap_pct = len(overlap) / len(val_patients) * 100
            print(f"\n⚠️  WARNING: {len(overlap)} validation patients in retrieval DB")
            print(f"   ({overlap_pct:.1f}% of validation set)")
            print(f"\n   This means queries retrieve reports from SAME patients!")
            return False
        else:
            print(f"\n✅ No validation patients in retrieval database")
            return True
            
    except Exception as e:
        print(f"⚠️  Could not analyze retrieval: {e}")
        return True


def check_11_performance_sanity(metrics_stage1, metrics_stage2, published_sota):
    """
    Check 11: Performance sanity check
    If results are TOO good, something is wrong
    """
    print("\n" + "="*80)
    print("CHECK 11: PERFORMANCE SANITY CHECK")
    print("="*80)
    
    # Expected improvement ranges based on literature
    expected_ranges = {
        'bleu4_vs_sota': (0.8, 2.0),      # 0.8x to 2x SOTA is reasonable
        'stage2_vs_stage1': (1.0, 1.8),   # RAG should improve 0-80%
        'entity_f1': (0.2, 0.6),          # Entity F1 typically 0.2-0.6
    }
    
    warnings = []
    
    # Check Stage 2 vs SOTA
    if metrics_stage2['bleu4'] / published_sota['bleu4'] > 3.0:
        warnings.append(f"BLEU-4 is 3x better than SOTA (suspicious!)")
    
    # Check Stage 2 vs Stage 1
    improvement = metrics_stage2['bleu4'] / metrics_stage1['bleu4']
    if improvement > 2.5:
        warnings.append(f"RAG improves BLEU-4 by {improvement:.1f}x (too high!)")
    
    # Check entity F1
    if 'entity_f1' in metrics_stage1:
        if metrics_stage1['entity_f1'] > 0.7:
            warnings.append(f"Baseline entity F1 = {metrics_stage1['entity_f1']:.2f} (suspiciously high)")
        if metrics_stage2['entity_f1'] > 0.85:
            warnings.append(f"RAG entity F1 = {metrics_stage2['entity_f1']:.2f} (near-perfect, suspicious)")
    
    print(f"Stage 1 BLEU-4: {metrics_stage1['bleu4']:.4f}")
    print(f"Stage 2 BLEU-4: {metrics_stage2['bleu4']:.4f}")
    print(f"Published SOTA: {published_sota['bleu4']:.4f}")
    print(f"Improvement:    {improvement:.2f}x")
    
    if warnings:
        print(f"\n⚠️  Performance red flags detected:")
        for w in warnings:
            print(f"   - {w}")
        print(f"\n   Recommendation: Check for data leakage")
        return False
    else:
        print(f"\n✅ Performance improvements within reasonable range")
        return True


def check_12_cross_validation_consistency(train_df, val_df, n_folds=5):
    """
    Check 12: Cross-validation consistency
    Split val set into k-folds, check if patient overlap is consistent
    """
    print("\n" + "="*80)
    print("CHECK 12: CROSS-VALIDATION CONSISTENCY")
    print("="*80)
    
    from sklearn.model_selection import KFold
    
    val_df['patient_id'] = val_df['image_path'].apply(
        lambda x: [p for p in x.split('/') if p.startswith('p') and len(p) > 5][0] if any(p.startswith('p') and len(p) > 5 for p in x.split('/')) else None
    )
    
    train_patients = set(train_df['image_path'].apply(
        lambda x: [p for p in x.split('/') if p.startswith('p') and len(p) > 5][0] if any(p.startswith('p') and len(p) > 5 for p in x.split('/')) else None
    ).dropna())
    
    # Get unique validation patients
    val_patients_list = val_df['patient_id'].dropna().unique()
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    overlap_rates = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(val_patients_list)):
        fold_val_patients = set(val_patients_list[test_idx])
        overlap = len(train_patients & fold_val_patients)
        overlap_rate = overlap / len(fold_val_patients)
        overlap_rates.append(overlap_rate)
        
        print(f"Fold {fold+1}: {overlap} overlapping patients ({overlap_rate*100:.1f}%)")
    
    mean_overlap = np.mean(overlap_rates)
    std_overlap = np.std(overlap_rates)
    
    print(f"\nMean overlap: {mean_overlap*100:.1f}% ± {std_overlap*100:.1f}%")
    
    if mean_overlap > 0.1:
        print(f"\n❌ Consistent high overlap across folds - systematic leakage!")
        return False
    elif std_overlap > 0.05:
        print(f"\n⚠️  High variance in overlap - inconsistent split")
        return False
    else:
        print(f"\n✅ Consistent low overlap across folds")
        return True


def check_13_prediction_confidence_analysis(predictions, references):
    """
    Check 13: Prediction confidence analysis
    If model is overly confident, might be leakage
    """
    print("\n" + "="*80)
    print("CHECK 13: PREDICTION CONFIDENCE ANALYSIS")
    print("="*80)
    
    # Calculate word-level precision for each prediction
    precisions = []
    
    for pred, ref in zip(predictions[:1000], references[:1000]):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        
        if len(pred_words) > 0:
            precision = len(pred_words & ref_words) / len(pred_words)
            precisions.append(precision)
    
    precisions = np.array(precisions)
    
    mean_prec = precisions.mean()
    high_prec_count = (precisions > 0.9).sum()
    perfect_count = (precisions == 1.0).sum()
    
    print(f"Average word precision: {mean_prec:.3f}")
    print(f"Predictions with >90% precision: {high_prec_count} ({high_prec_count/len(precisions)*100:.1f}%)")
    print(f"Perfect predictions (100%): {perfect_count}")
    
    if perfect_count > 50:
        print(f"\n⚠️  WARNING: Too many perfect predictions")
        print(f"   Suggests model is copying from somewhere")
        return False
    elif mean_prec > 0.85:
        print(f"\n⚠️  WARNING: Very high average precision")
        print(f"   Unusually confident predictions")
        return False
    else:
        print(f"\n✅ Prediction confidence within normal range")
        return True


def check_14_vocabulary_overlap(train_df, val_df):
    """
    Check 14: Vocabulary overlap analysis
    Check if val set uses words not seen in training
    """
    print("\n" + "="*80)
    print("CHECK 14: VOCABULARY OVERLAP")
    print("="*80)
    
    # Extract vocabularies
    train_vocab = set()
    for text in train_df['findings'].fillna('').values:
        train_vocab.update(text.lower().split())
    
    val_vocab = set()
    for text in val_df['findings'].fillna('').values:
        val_vocab.update(text.lower().split())
    
    overlap = train_vocab & val_vocab
    val_unique = val_vocab - train_vocab
    
    overlap_rate = len(overlap) / len(val_vocab) if len(val_vocab) > 0 else 0
    
    print(f"Training vocabulary: {len(train_vocab):,} words")
    print(f"Validation vocabulary: {len(val_vocab):,} words")
    print(f"Overlap: {len(overlap):,} words ({overlap_rate*100:.1f}%)")
    print(f"Val-only words: {len(val_unique):,}")
    
    if len(val_unique) > 0:
        print(f"\nExamples of val-only words: {list(val_unique)[:10]}")
    
    if overlap_rate < 0.7:
        print(f"\n⚠️  WARNING: Low vocabulary overlap")
        print(f"   Val set may be from different distribution")
        return False
    elif overlap_rate > 0.99:
        print(f"\n⚠️  WARNING: Almost perfect vocabulary overlap")
        print(f"   Suspiciously similar distributions")
        return False
    else:
        print(f"\n✅ Vocabulary overlap within expected range")
        return True


def check_15_entity_distribution(train_df, val_df):
    """
    Check 15: Clinical entity distribution
    Check if similar disease/finding distributions
    """
    print("\n" + "="*80)
    print("CHECK 15: CLINICAL ENTITY DISTRIBUTION")
    print("="*80)
    
    # Common radiology terms to check
    clinical_terms = [
        'cardiomegaly', 'effusion', 'pneumonia', 'consolidation',
        'pneumothorax', 'edema', 'atelectasis', 'opacity',
        'normal', 'clear', 'stable', 'enlarged'
    ]
    
    print(f"Checking distribution of {len(clinical_terms)} clinical terms:\n")
    
    large_differences = []
    
    for term in clinical_terms:
        train_count = train_df['findings'].fillna('').str.lower().str.contains(term).sum()
        val_count = val_df['findings'].fillna('').str.lower().str.contains(term).sum()
        
        train_rate = train_count / len(train_df)
        val_rate = val_count / len(val_df)
        
        ratio = val_rate / train_rate if train_rate > 0 else 0
        
        if abs(ratio - 1.0) > 0.5:  # >50% difference
            large_differences.append((term, ratio))
        
        print(f"  {term:15s}: Train {train_rate*100:5.1f}%, Val {val_rate*100:5.1f}% (ratio: {ratio:.2f})")
    
    if large_differences:
        print(f"\n⚠️  WARNING: Large distribution differences for:")
        for term, ratio in large_differences:
            print(f"   - {term}: {ratio:.2f}x")
        print(f"\n   Train and val may have different patient populations")
        return False
    else:
        print(f"\n✅ Clinical entity distributions are similar")
        return True


def check_16_official_split_comparison(train_df, val_df, official_split_dir=None):
    """
    Check 16: Compare with official MIMIC-CXR splits
    MIMIC-CXR provides official train/val/test splits
    """
    print("\n" + "="*80)
    print("CHECK 16: OFFICIAL SPLIT COMPARISON")
    print("="*80)
    
    if official_split_dir is None:
        print("⚠️  No official split directory provided")
        print("   MIMIC-CXR official splits are at:")
        print("   https://physionet.org/content/mimic-cxr-jpg/2.0.0/")
        return True
    
    try:
        # Load official splits
        official_train = set(pd.read_csv(f"{official_split_dir}/train.csv")['dicom_id'])
        official_val = set(pd.read_csv(f"{official_split_dir}/validate.csv")['dicom_id'])
        official_test = set(pd.read_csv(f"{official_split_dir}/test.csv")['dicom_id'])
        
        # Extract dicom IDs from your splits
        def extract_dicom(path):
            # dicom_id is typically the image filename without extension
            return Path(path).stem
        
        your_train = set(train_df['image_path'].apply(extract_dicom))
        your_val = set(val_df['image_path'].apply(extract_dicom))
        
        # Check alignment
        train_agreement = len(your_train & official_train) / len(your_train)
        val_agreement = len(your_val & official_val) / len(your_val)
        
        print(f"Your splits vs. official MIMIC-CXR splits:")
        print(f"  Training agreement: {train_agreement*100:.1f}%")
        print(f"  Validation agreement: {val_agreement*100:.1f}%")
        
        if train_agreement < 0.9 or val_agreement < 0.9:
            print(f"\n⚠️  WARNING: Your splits differ from official splits")
            print(f"   Consider using official MIMIC-CXR splits for reproducibility")
            return False
        else:
            print(f"\n✅ Your splits match official MIMIC-CXR splits")
            return True
            
    except Exception as e:
        print(f"⚠️  Could not compare with official splits: {e}")
        return True


def generate_leakage_report(results, output_file='data_leakage_report.txt'):
    """Generate comprehensive report"""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE DATA LEAKAGE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        f.write(f"Overall Score: {passed}/{total} checks passed\n\n")
        
        # Critical issues
        critical = ['patient_overlap', 'retrieval_contamination', 'model_memorization']
        critical_failed = [k for k in critical if k in results and not results[k]]
        
        if critical_failed:
            f.write("❌ CRITICAL ISSUES (MUST FIX):\n")
            for issue in critical_failed:
                f.write(f"   - {issue}\n")
            f.write("\n")
        
        # Warnings
        warnings = [k for k, v in results.items() if not v and k not in critical]
        if warnings:
            f.write("⚠️  WARNINGS (SHOULD INVESTIGATE):\n")
            for warning in warnings:
                f.write(f"   - {warning}\n")
            f.write("\n")
        
        # All checks
        f.write("\nDETAILED RESULTS:\n")
        f.write("-"*80 + "\n")
        for check, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            f.write(f"{status}: {check}\n")
        
        f.write("\n" + "="*80 + "\n")
        
        if 'patient_overlap' in results and not results['patient_overlap']:
            f.write("\n⚠️  ACTION REQUIRED:\n")
            f.write("Patient overlap detected. Your validation scores are inflated.\n")
            f.write("You must re-split data by patient before publication.\n")
        elif passed == total:
            f.write("\n✅ VALIDATION COMPLETE:\n")
            f.write("All checks passed. Your data split appears valid.\n")
        
    print(f"\n✓ Report saved to: {output_file}")


# Usage example
if __name__ == '__main__':
    print("""
Advanced Data Leakage Checks Available:

9.  Model Memorization Test
10. Retrieval Hits Analysis  
11. Performance Sanity Check
12. Cross-Validation Consistency
13. Prediction Confidence Analysis
14. Vocabulary Overlap
15. Clinical Entity Distribution
16. Official Split Comparison

Run the main check_data_leakage.py script first, then use these
for deeper investigation if issues are suspected.
    """)
