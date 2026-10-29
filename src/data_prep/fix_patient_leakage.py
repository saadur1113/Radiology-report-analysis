"""
Fix Patient-Level Data Leakage - SIMPLIFIED VERSION
Re-split MIMIC-CXR at patient level to ensure no overlap
"""

import pandas as pd
import numpy as np
from pathlib import Path


def extract_patient_id(image_path):
    """Extract patient ID from MIMIC-CXR path"""
    parts = str(image_path).split('/')
    for part in parts:
        if part.startswith('p') and len(part) > 5:
            return part
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix patient-level data leakage')
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./data/processed')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print("="*80)
    print("FIXING PATIENT-LEVEL DATA LEAKAGE")
    print("="*80)
    
    # STEP 1: Load and combine data
    print("\nSTEP 1: Loading data...")
    
    dfs = []
    train_orig = pd.read_csv(args.train_csv)
    print(f"  Train: {len(train_orig):,} samples")
    dfs.append(train_orig)
    
    val_orig = pd.read_csv(args.val_csv)
    print(f"  Val:   {len(val_orig):,} samples")
    dfs.append(val_orig)
    
    if args.test_csv:
        test_orig = pd.read_csv(args.test_csv)
        print(f"  Test:  {len(test_orig):,} samples")
        dfs.append(test_orig)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined):,} samples")
    
    # STEP 2: Extract patient IDs
    print("\nSTEP 2: Extracting patient IDs...")
    combined['patient_id'] = combined['image_path'].apply(extract_patient_id)
    
    # Remove invalid
    original_len = len(combined)
    combined = combined[combined['patient_id'].notna()]
    if len(combined) < original_len:
        print(f"  Removed {original_len - len(combined)} samples without valid patient ID")
    
    # Get unique patients
    unique_patients = combined['patient_id'].unique()
    print(f"  Found {len(unique_patients):,} unique patients")
    
    # Patient statistics
    patient_counts = combined.groupby('patient_id').size()
    print(f"\n  Patient statistics:")
    print(f"    Mean images/patient: {patient_counts.mean():.1f}")
    print(f"    Median: {patient_counts.median():.0f}")
    print(f"    Max: {patient_counts.max()}")
    
    # STEP 3: Split patients
    print(f"\nSTEP 3: Splitting by patient (seed={args.seed})...")
    
    np.random.seed(args.seed)
    shuffled_patients = np.random.permutation(unique_patients)
    
    n_patients = len(shuffled_patients)
    n_train = int(args.train_ratio * n_patients)
    n_val = int(args.val_ratio * n_patients)
    
    train_patients = set(shuffled_patients[:n_train])
    val_patients = set(shuffled_patients[n_train:n_train+n_val])
    test_patients = set(shuffled_patients[n_train+n_val:])
    
    print(f"\n  Patient split:")
    print(f"    Train: {len(train_patients):,} ({len(train_patients)/n_patients*100:.1f}%)")
    print(f"    Val:   {len(val_patients):,} ({len(val_patients)/n_patients*100:.1f}%)")
    print(f"    Test:  {len(test_patients):,} ({len(test_patients)/n_patients*100:.1f}%)")
    
    # Verify no overlap
    assert len(train_patients & val_patients) == 0, "Train-Val overlap!"
    assert len(train_patients & test_patients) == 0, "Train-Test overlap!"
    assert len(val_patients & test_patients) == 0, "Val-Test overlap!"
    print(f"  ✓ Verified: No patient overlap")
    
    # STEP 4: Create sample splits
    print("\nSTEP 4: Creating sample splits...")
    
    train_clean = combined[combined['patient_id'].isin(train_patients)].drop(columns=['patient_id']).copy()
    val_clean = combined[combined['patient_id'].isin(val_patients)].drop(columns=['patient_id']).copy()
    test_clean = combined[combined['patient_id'].isin(test_patients)].drop(columns=['patient_id']).copy()
    
    print(f"\n  Sample split:")
    print(f"    Train: {len(train_clean):,} ({len(train_clean)/len(combined)*100:.1f}%)")
    print(f"    Val:   {len(val_clean):,} ({len(val_clean)/len(combined)*100:.1f}%)")
    print(f"    Test:  {len(test_clean):,} ({len(test_clean)/len(combined)*100:.1f}%)")
    
    # STEP 5: Save
    print("\nSTEP 5: Saving clean splits...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'mimic_cxr_train_clean.csv'
    val_path = output_dir / 'mimic_cxr_val_clean.csv'
    test_path = output_dir / 'mimic_cxr_test_clean.csv'
    
    train_clean.to_csv(train_path, index=False)
    print(f"  ✓ {train_path}")
    
    val_clean.to_csv(val_path, index=False)
    print(f"  ✓ {val_path}")
    
    test_clean.to_csv(test_path, index=False)
    print(f"  ✓ {test_path}")
    
    # STEP 6: Verify
    print("\nSTEP 6: Final verification...")
    
    train_clean['patient_id'] = train_clean['image_path'].apply(extract_patient_id)
    val_clean['patient_id'] = val_clean['image_path'].apply(extract_patient_id)
    test_clean['patient_id'] = test_clean['image_path'].apply(extract_patient_id)
    
    train_pts = set(train_clean['patient_id'].dropna())
    val_pts = set(val_clean['patient_id'].dropna())
    test_pts = set(test_clean['patient_id'].dropna())
    
    overlap_train_val = len(train_pts & val_pts)
    overlap_train_test = len(train_pts & test_pts)
    overlap_val_test = len(val_pts & test_pts)
    
    print(f"\n  Patient overlap check:")
    print(f"    Train ∩ Val:  {overlap_train_val}")
    print(f"    Train ∩ Test: {overlap_train_test}")
    print(f"    Val ∩ Test:   {overlap_val_test}")
    
    if overlap_train_val == 0 and overlap_train_test == 0 and overlap_val_test == 0:
        print(f"\n  ✅ SUCCESS: Perfect patient-level separation!")
    else:
        print(f"\n  ❌ ERROR: Overlap still detected!")
        return 1
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nClean splits created:")
    print(f"  {train_path}")
    print(f"    → {len(train_clean):,} samples, {len(train_pts):,} patients")
    print(f"  {val_path}")
    print(f"    → {len(val_clean):,} samples, {len(val_pts):,} patients")
    print(f"  {test_path}")
    print(f"    → {len(test_clean):,} samples, {len(test_pts):,} patients")
    
    print(f"\nNext steps:")
    print(f"  1. Rebuild retrieval:")
    print(f"     python build_retrieval_index.py --csv_file {train_path}")
    print(f"\n  2. Retrain models:")
    print(f"     python train_stage1_opt.py --train_csv {train_path} --val_csv {val_path}")
    print(f"     python train_stage2_opt.py --train_csv {train_path} --val_csv {val_path}")
    
    print("\n" + "="*80)
    print("✅ DONE!")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    exit(main())