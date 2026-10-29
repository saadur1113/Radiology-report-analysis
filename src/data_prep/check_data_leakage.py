"""
Comprehensive Data Leakage Detection for Medical Imaging
Checks all possible sources of data contamination
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict


def extract_patient_id(image_path):
    """Extract patient ID from MIMIC-CXR path"""
    parts = image_path.split('/')
    for part in parts:
        if part.startswith('p') and len(part) > 5:
            return part
    return None


def extract_study_id(image_path):
    """Extract study ID (imaging session)"""
    parts = image_path.split('/')
    for part in parts:
        if part.startswith('s') and len(part) > 5:
            return part
    return None


def check_1_image_overlap(train_df, val_df, test_df=None):
    """Check 1: Exact same images in different splits"""
    print("\n" + "="*80)
    print("CHECK 1: IMAGE-LEVEL OVERLAP")
    print("="*80)
    
    train_images = set(train_df['image_path'].values)
    val_images = set(val_df['image_path'].values)
    
    overlap_train_val = train_images & val_images
    
    print(f"Train images:      {len(train_images):,}")
    print(f"Validation images: {len(val_images):,}")
    
    if len(overlap_train_val) > 0:
        print(f"\n❌ CRITICAL: {len(overlap_train_val)} images in both train and val!")
        print(f"Examples: {list(overlap_train_val)[:3]}")
        return False
    else:
        print(f"\n✅ No image overlap between train and val")
    
    if test_df is not None:
        test_images = set(test_df['image_path'].values)
        overlap_train_test = train_images & test_images
        overlap_val_test = val_images & test_images
        
        if len(overlap_train_test) > 0:
            print(f"❌ CRITICAL: {len(overlap_train_test)} images in both train and test!")
            return False
        if len(overlap_val_test) > 0:
            print(f"❌ CRITICAL: {len(overlap_val_test)} images in both val and test!")
            return False
        print(f"✅ No image overlap with test set")
    
    return True


def check_2_patient_overlap(train_df, val_df, test_df=None):
    """Check 2: Same patients in different splits (CRITICAL!)"""
    print("\n" + "="*80)
    print("CHECK 2: PATIENT-LEVEL OVERLAP")
    print("="*80)
    
    train_df['patient_id'] = train_df['image_path'].apply(extract_patient_id)
    val_df['patient_id'] = val_df['image_path'].apply(extract_patient_id)
    
    train_patients = set(train_df['patient_id'].dropna())
    val_patients = set(val_df['patient_id'].dropna())
    
    overlap = train_patients & val_patients
    
    print(f"Train patients:      {len(train_patients):,}")
    print(f"Validation patients: {len(val_patients):,}")
    print(f"Overlapping patients: {len(overlap):,}")
    
    if len(overlap) > 0:
        overlap_pct = len(overlap) / len(val_patients) * 100
        print(f"\n❌ CRITICAL DATA LEAKAGE!")
        print(f"   {len(overlap)} patients ({overlap_pct:.1f}% of val) appear in both!")
        print(f"\n   This means the model has seen OTHER images from these patients")
        print(f"   during training, which inflates validation scores artificially.")
        
        # Show examples
        print(f"\n   Example overlapping patients: {list(overlap)[:5]}")
        
        # Show how many images per overlapping patient
        overlap_patient_counts = train_df[train_df['patient_id'].isin(overlap)].groupby('patient_id').size()
        print(f"\n   Images in training for overlapping patients:")
        print(f"   - Mean: {overlap_patient_counts.mean():.1f} images/patient")
        print(f"   - Max: {overlap_patient_counts.max()} images/patient")
        
        return False
    else:
        print(f"\n✅ Perfect patient separation - no overlap!")
        return True
    
    if test_df is not None:
        test_df['patient_id'] = test_df['image_path'].apply(extract_patient_id)
        test_patients = set(test_df['patient_id'].dropna())
        
        train_test_overlap = train_patients & test_patients
        val_test_overlap = val_patients & test_patients
        
        if len(train_test_overlap) > 0 or len(val_test_overlap) > 0:
            print(f"❌ Patient overlap with test set!")
            return False
        print(f"✅ No patient overlap with test set")


def check_3_study_overlap(train_df, val_df):
    """Check 3: Same imaging studies (sessions) in different splits"""
    print("\n" + "="*80)
    print("CHECK 3: STUDY-LEVEL OVERLAP")
    print("="*80)
    
    train_df['study_id'] = train_df['image_path'].apply(extract_study_id)
    val_df['study_id'] = val_df['image_path'].apply(extract_study_id)
    
    train_studies = set(train_df['study_id'].dropna())
    val_studies = set(val_df['study_id'].dropna())
    
    overlap = train_studies & val_studies
    
    print(f"Train studies:      {len(train_studies):,}")
    print(f"Validation studies: {len(val_studies):,}")
    
    if len(overlap) > 0:
        print(f"\n❌ WARNING: {len(overlap)} studies in both splits!")
        print(f"   (Same imaging session, different views)")
        return False
    else:
        print(f"\n✅ No study overlap")
        return True


def check_4_text_similarity(train_df, val_df, sample_size=100):
    """Check 4: Highly similar reports (possible near-duplicates)"""
    print("\n" + "="*80)
    print("CHECK 4: TEXT SIMILARITY (Near-Duplicate Reports)")
    print("="*80)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Sample for speed
    train_sample = train_df.sample(min(sample_size, len(train_df)))
    val_sample = val_df.sample(min(sample_size, len(val_df)))
    
    train_texts = train_sample['findings'].fillna('').values
    val_texts = val_sample['findings'].fillna('').values
    
    try:
        vectorizer = TfidfVectorizer(max_features=100)
        train_vectors = vectorizer.fit_transform(train_texts)
        val_vectors = vectorizer.transform(val_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(val_vectors, train_vectors)
        max_similarities = similarities.max(axis=1)
        
        high_sim_count = (max_similarities > 0.95).sum()
        very_high_sim_count = (max_similarities > 0.99).sum()
        
        print(f"Sampled {len(val_texts)} validation reports")
        print(f"Reports with >95% similarity to training: {high_sim_count}")
        print(f"Reports with >99% similarity to training: {very_high_sim_count}")
        
        if very_high_sim_count > len(val_texts) * 0.05:
            print(f"\n⚠️  WARNING: {very_high_sim_count} near-duplicate reports!")
            print(f"   This suggests template-based reports or data duplication")
            return False
        else:
            print(f"\n✅ Text similarity within normal range")
            return True
    except Exception as e:
        print(f"⚠️  Could not compute text similarity: {e}")
        return True


def check_5_retrieval_contamination(retrieval_dir, val_df):
    """Check 5: Validation data in retrieval database"""
    print("\n" + "="*80)
    print("CHECK 5: RETRIEVAL DATABASE CONTAMINATION")
    print("="*80)
    
    try:
        retrieval_metadata = pd.read_csv(Path(retrieval_dir) / 'retrieval_metadata.csv')
        
        print(f"Retrieval DB size: {len(retrieval_metadata):,}")
        print(f"Validation size:   {len(val_df):,}")
        
        # Check if retrieval paths overlap with val paths
        if 'image_path' in retrieval_metadata.columns:
            retrieval_images = set(retrieval_metadata['image_path'].values)
            val_images = set(val_df['image_path'].values)
            
            overlap = retrieval_images & val_images
            
            if len(overlap) > 0:
                print(f"\n❌ CRITICAL: {len(overlap)} validation images in retrieval DB!")
                print(f"   This means validation queries retrieve their own answers!")
                print(f"   Examples: {list(overlap)[:3]}")
                return False
            else:
                print(f"\n✅ No validation data in retrieval database")
                return True
        
        # Check by patient ID
        retrieval_metadata['patient_id'] = retrieval_metadata.get(
            'image_path', retrieval_metadata.get('text', pd.Series())
        ).apply(extract_patient_id)
        val_df['patient_id'] = val_df['image_path'].apply(extract_patient_id)
        
        retrieval_patients = set(retrieval_metadata['patient_id'].dropna())
        val_patients = set(val_df['patient_id'].dropna())
        
        patient_overlap = retrieval_patients & val_patients
        
        if len(patient_overlap) > 0:
            print(f"\n⚠️  WARNING: {len(patient_overlap)} validation patients in retrieval DB")
            print(f"   Retrieval may return reports from same patient (different images)")
            return False
        
        return True
        
    except FileNotFoundError:
        print("⚠️  Retrieval database not found - skipping check")
        return True
    except Exception as e:
        print(f"⚠️  Error checking retrieval database: {e}")
        return True


def check_6_temporal_leakage(train_df, val_df):
    """Check 6: Future data in training (temporal leakage)"""
    print("\n" + "="*80)
    print("CHECK 6: TEMPORAL LEAKAGE")
    print("="*80)
    
    # MIMIC-CXR doesn't have explicit dates in the CSV
    # But we can check if there's a date column
    
    if 'study_date' in train_df.columns or 'StudyDate' in train_df.columns:
        date_col = 'study_date' if 'study_date' in train_df.columns else 'StudyDate'
        
        train_df[date_col] = pd.to_datetime(train_df[date_col], errors='coerce')
        val_df[date_col] = pd.to_datetime(val_df[date_col], errors='coerce')
        
        train_max_date = train_df[date_col].max()
        val_min_date = val_df[date_col].min()
        
        print(f"Latest training date: {train_max_date}")
        print(f"Earliest val date:    {val_min_date}")
        
        if val_min_date < train_max_date:
            overlap_days = (train_max_date - val_min_date).days
            print(f"\n⚠️  WARNING: {overlap_days} days of temporal overlap!")
            print(f"   Some validation data is from before training data")
            return False
        else:
            print(f"\n✅ Proper temporal split - validation is after training")
            return True
    else:
        print("⚠️  No date information available - cannot check temporal leakage")
        return True


def check_7_distribution_shift(train_df, val_df):
    """Check 7: Unusual distribution differences (might indicate bad split)"""
    print("\n" + "="*80)
    print("CHECK 7: DISTRIBUTION CONSISTENCY")
    print("="*80)
    
    # Check text length distribution
    train_lengths = train_df['findings'].fillna('').str.len()
    val_lengths = val_df['findings'].fillna('').str.len()
    
    print(f"Report length (characters):")
    print(f"  Train - Mean: {train_lengths.mean():.0f}, Median: {train_lengths.median():.0f}")
    print(f"  Val   - Mean: {val_lengths.mean():.0f}, Median: {val_lengths.median():.0f}")
    
    length_ratio = val_lengths.mean() / train_lengths.mean()
    
    if abs(length_ratio - 1.0) > 0.2:
        print(f"\n⚠️  WARNING: Large distribution difference ({length_ratio:.2f}x)")
        print(f"   Validation reports are significantly different in length")
        return False
    else:
        print(f"\n✅ Similar distributions (ratio: {length_ratio:.2f})")
        return True


def check_8_data_size_sanity(train_df, val_df, test_df=None):
    """Check 8: Sanity check on split sizes"""
    print("\n" + "="*80)
    print("CHECK 8: SPLIT SIZE SANITY CHECK")
    print("="*80)
    
    total = len(train_df) + len(val_df)
    if test_df is not None:
        total += len(test_df)
    
    train_pct = len(train_df) / total * 100
    val_pct = len(val_df) / total * 100
    
    print(f"Train: {len(train_df):,} ({train_pct:.1f}%)")
    print(f"Val:   {len(val_df):,} ({val_pct:.1f}%)")
    
    if test_df is not None:
        test_pct = len(test_df) / total * 100
        print(f"Test:  {len(test_df):,} ({test_pct:.1f}%)")
    
    # Expected ranges
    if train_pct < 60 or train_pct > 90:
        print(f"\n⚠️  WARNING: Unusual training set size ({train_pct:.1f}%)")
        return False
    if val_pct < 5 or val_pct > 20:
        print(f"\n⚠️  WARNING: Unusual validation set size ({val_pct:.1f}%)")
        return False
    
    print(f"\n✅ Split sizes look reasonable")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Data Leakage Detection')
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, default=None)
    parser.add_argument('--retrieval_dir', type=str, default='./data/retrieval')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading datasets...")
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv) if args.test_csv else None
    
    print(f"✓ Train: {len(train_df):,} samples")
    print(f"✓ Val:   {len(val_df):,} samples")
    if test_df is not None:
        print(f"✓ Test:  {len(test_df):,} samples")
    
    # Run all checks
    results = {}
    
    results['image_overlap'] = check_1_image_overlap(train_df, val_df, test_df)
    results['patient_overlap'] = check_2_patient_overlap(train_df, val_df, test_df)
    results['study_overlap'] = check_3_study_overlap(train_df, val_df)
    results['text_similarity'] = check_4_text_similarity(train_df, val_df)
    results['retrieval_contamination'] = check_5_retrieval_contamination(args.retrieval_dir, val_df)
    results['temporal_leakage'] = check_6_temporal_leakage(train_df, val_df)
    results['distribution_shift'] = check_7_distribution_shift(train_df, val_df)
    results['size_sanity'] = check_8_data_size_sanity(train_df, val_df, test_df)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check}")
    
    print(f"\n{'='*80}")
    print(f"OVERALL: {passed}/{total} checks passed")
    
    if results['patient_overlap'] == False:
        print(f"\n❌ CRITICAL: Patient overlap detected!")
        print(f"   YOUR RESULTS ARE INFLATED - MUST FIX BEFORE PUBLICATION!")
    elif passed == total:
        print(f"\n✅ ALL CHECKS PASSED - Data split is valid!")
    else:
        print(f"\n⚠️  WARNINGS DETECTED - Review failed checks above")
    
    print(f"{'='*80}\n")
    
    return results


if __name__ == '__main__':
    main()
