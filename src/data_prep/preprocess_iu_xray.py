"""
Preprocess IU X-Ray Dataset for Evaluation

Converts the Kaggle IU X-Ray download into the same CSV format
as MIMIC-CXR (image_path, findings) for direct use with our
evaluation scripts.

Steps:
1. Load reports and projections CSVs
2. Match frontal images to reports (skip lateral — matching MIMIC-CXR which uses frontal)
3. Download R2Gen standard train/val/test split (or create 70/10/20 split)
4. Output: iu_xray_test_clean.csv with columns: image_path, findings

Usage:
    python preprocess_iu_xray.py \
        --data_dir ./data/iu_xray \
        --output_dir ./data/processed
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='./data/iu_xray')
    parser.add_argument('--output_dir', type=str,
                        default='./data/processed')
    parser.add_argument('--r2gen_annotation', type=str, default=None,
                        help='Path to R2Gen annotation.json for standard split. If not provided, uses 70/10/20 random split.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("PREPROCESSING IU X-RAY DATASET")
    print("=" * 70)

    # Load CSVs
    reports_path = os.path.join(args.data_dir, 'indiana_reports.csv')
    projections_path = os.path.join(args.data_dir, 'indiana_projections.csv')
    images_dir = os.path.join(args.data_dir, 'images', 'images_normalized')

    reports = pd.read_csv(reports_path)
    projections = pd.read_csv(projections_path)

    print(f"\n[1/5] Loaded data:")
    print(f"  Reports: {len(reports)} studies")
    print(f"  Projections: {len(projections)} images")

    # Filter to frontal images only (matching MIMIC-CXR which uses frontal views)
    frontal = projections[projections['projection'] == 'Frontal'].copy()
    print(f"\n[2/5] Frontal images: {len(frontal)}")

    # For studies with multiple frontal images, keep the first one
    frontal = frontal.drop_duplicates(subset='uid', keep='first')
    print(f"  After dedup (one per study): {len(frontal)}")

    # Merge with reports
    merged = frontal.merge(reports[['uid', 'findings', 'impression']], on='uid', how='inner')
    print(f"\n[3/5] Merged (frontal images with reports): {len(merged)}")

    # Build full image paths
    merged['image_path'] = merged['filename'].apply(
        lambda f: os.path.join(images_dir, f)
    )

    # Verify images exist
    merged['image_exists'] = merged['image_path'].apply(os.path.exists)
    missing = merged[~merged['image_exists']]
    if len(missing) > 0:
        print(f"  WARNING: {len(missing)} images not found on disk")
        merged = merged[merged['image_exists']]

    # Clean findings text
    # IU X-Ray uses "XXXX" as a de-identification placeholder
    merged['findings'] = merged['findings'].astype(str).str.strip()
    merged['impression'] = merged['impression'].astype(str).str.strip()

    # Remove rows with empty or nan findings
    merged = merged[merged['findings'].notna()]
    merged = merged[merged['findings'] != 'nan']
    merged = merged[merged['findings'].str.len() > 10]  # Skip very short/empty

    print(f"  After filtering empty findings: {len(merged)}")

    # Try to use R2Gen standard split
    r2gen_path = args.r2gen_annotation
    if r2gen_path is None:
        # Try to find it in the data dir
        candidate = os.path.join(args.data_dir, 'annotation.json')
        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            r2gen_path = candidate

    if r2gen_path and os.path.exists(r2gen_path) and os.path.getsize(r2gen_path) > 0:
        print(f"\n[4/5] Using R2Gen standard split from {r2gen_path}")
        with open(r2gen_path, 'r') as f:
            r2gen = json.load(f)

        # Extract UIDs for each split from R2Gen
        # R2Gen uses image filenames like "CXR1_1_IM-0001-4001.dcm.png"
        # We need to map these back to UIDs
        split_uids = {}
        for split in ['train', 'val', 'test']:
            if split in r2gen:
                uids = set()
                for entry in r2gen[split]:
                    # Try to extract UID from image path
                    img_paths = entry.get('image_path', [])
                    study_id = entry.get('id', '')
                    if study_id:
                        uids.add(str(study_id))
                    for img_p in img_paths:
                        # Extract UID from filename like "CXR1234_..."
                        parts = os.path.basename(img_p).split('_')
                        if parts:
                            uid = parts[0].replace('CXR', '')
                            uids.add(uid)
                split_uids[split] = uids
                print(f"  {split}: {len(uids)} studies")

        # Assign splits
        merged['split'] = 'unknown'
        for split, uids in split_uids.items():
            mask = merged['uid'].astype(str).isin(uids)
            merged.loc[mask, 'split'] = split

        unknown = (merged['split'] == 'unknown').sum()
        if unknown > 0:
            print(f"  WARNING: {unknown} studies not in R2Gen split, assigning to test")
            merged.loc[merged['split'] == 'unknown', 'split'] = 'test'
    else:
        print(f"\n[4/5] R2Gen annotation not found, using random 70/10/20 split")
        np.random.seed(42)
        n = len(merged)
        indices = np.random.permutation(n)
        train_end = int(0.7 * n)
        val_end = int(0.8 * n)

        merged = merged.reset_index(drop=True)
        merged['split'] = 'test'  # default
        merged.loc[indices[:train_end], 'split'] = 'train'
        merged.loc[indices[train_end:val_end], 'split'] = 'val'

    # Print split statistics
    print(f"\n  Split distribution:")
    for split in ['train', 'val', 'test']:
        count = (merged['split'] == split).sum()
        print(f"    {split}: {count} samples")

    # Save CSVs for each split
    print(f"\n[5/5] Saving CSVs...")
    for split in ['train', 'val', 'test']:
        split_df = merged[merged['split'] == split][['image_path', 'findings']].copy()
        split_df = split_df.reset_index(drop=True)

        output_path = os.path.join(args.output_dir, f'iu_xray_{split}_clean.csv')
        split_df.to_csv(output_path, index=False)
        print(f"  {split}: {len(split_df)} samples → {output_path}")

    # Also save full dataset (all splits)
    all_df = merged[['image_path', 'findings', 'split']].copy()
    all_path = os.path.join(args.output_dir, 'iu_xray_all.csv')
    all_df.to_csv(all_path, index=False)
    print(f"  all:  {len(all_df)} samples → {all_path}")

    # Print sample entries
    print(f"\n  Sample entries from test set:")
    test_df = merged[merged['split'] == 'test']
    for i in range(min(3, len(test_df))):
        row = test_df.iloc[i]
        print(f"    [{i}] {row['image_path']}")
        print(f"        {row['findings'][:100]}...")
        print()

    # Print comparison with MIMIC-CXR
    print(f"{'=' * 70}")
    print("COMPARISON WITH MIMIC-CXR")
    print(f"{'=' * 70}")
    print(f"  {'':20s} {'MIMIC-CXR':>12s} {'IU X-Ray':>12s}")
    print(f"  {'Train':20s} {'83,507':>12s} {(merged['split']=='train').sum():>12,d}")
    print(f"  {'Val':20s} {'11,822':>12s} {(merged['split']=='val').sum():>12,d}")
    print(f"  {'Test':20s} {'1,000 (eval)':>12s} {(merged['split']=='test').sum():>12,d}")
    print(f"{'=' * 70}")

    # Findings length comparison
    test_findings = test_df['findings'].astype(str)
    avg_len = test_findings.apply(lambda x: len(x.split())).mean()
    print(f"\n  IU X-Ray test avg findings length: {avg_len:.0f} words")
    print(f"  MIMIC-CXR val avg findings length: ~49 words")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
