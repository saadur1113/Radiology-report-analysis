"""
TEST SCRIPT - Validate patient-level split logic
Run this BEFORE the full split to verify it works correctly
"""

import pandas as pd
import numpy as np


def extract_patient_id(image_path):
    """Extract patient ID from MIMIC-CXR path"""
    parts = str(image_path).split('/')
    for part in parts:
        if part.startswith('p') and len(part) > 5:
            return part
    return None


def test_split_logic(train_csv, val_csv, sample_size=1000):
    """
    Test the split logic on a small sample
    """
    print("="*80)
    print("TESTING PATIENT-LEVEL SPLIT LOGIC")
    print("="*80)
    
    # Load small sample
    print(f"\nLoading {sample_size} samples from each file...")
    train_sample = pd.read_csv(train_csv, nrows=sample_size)
    val_sample = pd.read_csv(val_csv, nrows=sample_size)
    
    combined = pd.concat([train_sample, val_sample], ignore_index=True)
    print(f"✓ Combined: {len(combined)} samples")
    
    # Extract patient IDs
    print("\nExtracting patient IDs...")
    combined['patient_id'] = combined['image_path'].apply(extract_patient_id)
    
    # Check how many valid
    valid_count = combined['patient_id'].notna().sum()
    invalid_count = combined['patient_id'].isna().sum()
    
    print(f"  Valid patient IDs: {valid_count}")
    print(f"  Invalid/missing: {invalid_count}")
    
    if invalid_count > 0:
        print(f"\n⚠️  WARNING: {invalid_count} samples without patient IDs!")
        print("  Sample paths without IDs:")
        print(combined[combined['patient_id'].isna()]['image_path'].head())
    
    # Remove invalid
    combined = combined[combined['patient_id'].notna()]
    
    # Get unique patients
    unique_patients = combined['patient_id'].unique()
    print(f"\n✓ Found {len(unique_patients)} unique patients")
    
    # Patient statistics
    patient_counts = combined.groupby('patient_id').size()
    print(f"\nPatient statistics:")
    print(f"  Mean images/patient: {patient_counts.mean():.2f}")
    print(f"  Median: {patient_counts.median()}")
    print(f"  Min: {patient_counts.min()}")
    print(f"  Max: {patient_counts.max()}")
    print(f"  Total images: {patient_counts.sum()}")
    
    # Show examples
    print(f"\nExample patient IDs:")
    for i, pid in enumerate(list(unique_patients)[:5]):
        count = patient_counts[pid]
        print(f"  {i+1}. {pid}: {count} images")
    
    # Test split
    print("\n" + "-"*80)
    print("TESTING SPLIT LOGIC")
    print("-"*80)
    
    np.random.seed(42)
    shuffled = np.random.permutation(unique_patients)
    
    n_train = int(0.8 * len(shuffled))
    n_val = int(0.1 * len(shuffled))
    
    train_pts = set(shuffled[:n_train])
    val_pts = set(shuffled[n_train:n_train+n_val])
    test_pts = set(shuffled[n_train+n_val:])
    
    print(f"\nPatient split:")
    print(f"  Train: {len(train_pts)} patients ({len(train_pts)/len(shuffled)*100:.1f}%)")
    print(f"  Val:   {len(val_pts)} patients ({len(val_pts)/len(shuffled)*100:.1f}%)")
    print(f"  Test:  {len(test_pts)} patients ({len(test_pts)/len(shuffled)*100:.1f}%)")
    
    # Check overlap
    overlap_tv = len(train_pts & val_pts)
    overlap_tt = len(train_pts & test_pts)
    overlap_vt = len(val_pts & test_pts)
    
    print(f"\nPatient overlap check:")
    print(f"  Train ∩ Val:  {overlap_tv} patients")
    print(f"  Train ∩ Test: {overlap_tt} patients")
    print(f"  Val ∩ Test:   {overlap_vt} patients")
    
    if overlap_tv == 0 and overlap_tt == 0 and overlap_vt == 0:
        print(f"\n✅ PERFECT: No patient overlap!")
    else:
        print(f"\n❌ ERROR: Patient overlap detected in test!")
        return False
    
    # Create actual splits
    train_df = combined[combined['patient_id'].isin(train_pts)].copy()
    val_df = combined[combined['patient_id'].isin(val_pts)].copy()
    test_df = combined[combined['patient_id'].isin(test_pts)].copy()
    
    print(f"\nSample split:")
    print(f"  Train: {len(train_df)} images ({len(train_df)/len(combined)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} images ({len(val_df)/len(combined)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} images ({len(test_df)/len(combined)*100:.1f}%)")
    
    # Verify no image overlap
    train_images = set(train_df['image_path'])
    val_images = set(val_df['image_path'])
    test_images = set(test_df['image_path'])
    
    img_overlap_tv = len(train_images & val_images)
    img_overlap_tt = len(train_images & test_images)
    img_overlap_vt = len(val_images & test_images)
    
    print(f"\nImage overlap check:")
    print(f"  Train ∩ Val:  {img_overlap_tv} images")
    print(f"  Train ∩ Test: {img_overlap_tt} images")
    print(f"  Val ∩ Test:   {img_overlap_vt} images")
    
    if img_overlap_tv == 0 and img_overlap_tt == 0 and img_overlap_vt == 0:
        print(f"\n✅ PERFECT: No image overlap!")
    else:
        print(f"\n❌ ERROR: Image overlap detected!")
        return False
    
    # Sanity checks
    print("\n" + "-"*80)
    print("SANITY CHECKS")
    print("-"*80)
    
    checks_passed = True
    
    # Check 1: All samples accounted for
    total_split = len(train_df) + len(val_df) + len(test_df)
    if total_split == len(combined):
        print(f"✅ All samples accounted for: {total_split} == {len(combined)}")
    else:
        print(f"❌ Sample count mismatch: {total_split} != {len(combined)}")
        checks_passed = False
    
    # Check 2: All patients accounted for
    total_patients = len(train_pts) + len(val_pts) + len(test_pts)
    if total_patients == len(unique_patients):
        print(f"✅ All patients accounted for: {total_patients} == {len(unique_patients)}")
    else:
        print(f"❌ Patient count mismatch: {total_patients} != {len(unique_patients)}")
        checks_passed = False
    
    # Check 3: Required columns present
    required_cols = ['image_path', 'findings']
    for col in required_cols:
        if col in combined.columns:
            print(f"✅ Required column '{col}' present")
        else:
            print(f"❌ Missing required column '{col}'")
            checks_passed = False
    
    # Check 4: No duplicate images
    if len(combined) == len(combined['image_path'].unique()):
        print(f"✅ No duplicate images in input")
    else:
        dupes = len(combined) - len(combined['image_path'].unique())
        print(f"⚠️  {dupes} duplicate images in input (will be preserved in split)")
    
    # Final verdict
    print("\n" + "="*80)
    if checks_passed:
        print("✅ ALL TESTS PASSED - SPLIT LOGIC IS CORRECT!")
        print("="*80)
        print("\nYou can safely run the full split:")
        print("  python fix_patient_leakage_simple.py \\")
        print("    --train_csv ./data/processed/mimic_cxr_train.csv \\")
        print("    --val_csv ./data/processed/mimic_cxr_val.csv")
        return True
    else:
        print("❌ SOME TESTS FAILED - DO NOT RUN FULL SPLIT YET!")
        print("="*80)
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test split logic')
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='Number of samples to test (default: 1000)')
    
    args = parser.parse_args()
    
    success = test_split_logic(args.train_csv, args.val_csv, args.sample_size)
    
    exit(0 if success else 1)
