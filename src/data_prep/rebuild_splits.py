"""
Rebuild the exact train/val/test CSVs used in the paper from the ID-only
split files in ./splits and your own local copies of the datasets.

The split files contain identifiers only (no report text, no paths), so you
must first produce the full per-image tables with the preprocessing scripts:

  MIMIC-CXR : src/data_prep/prepare_mimic_data_separate_folders.py
              -> <processed_dir>/mimic_cxr_processed.csv
  IU X-Ray  : src/data_prep/preprocess_iu_xray.py
              -> <processed_dir>/iu_xray_all.csv
  CheXpert+ : src/data_prep/preprocess_chexpert_plus.py
              -> <processed_dir>/chexpert_plus_test_clean.csv  (already final;
                 this script only checks it against the split file)

Row order follows the split files. This matters: several evaluations use the
first 1,000 rows of mimic_cxr_val_clean.csv.

Usage:
    python src/data_prep/rebuild_splits.py --processed_dir ./data/processed
"""
import argparse
import os

import pandas as pd


def rebuild_mimic(processed_dir, splits_dir):
    src = os.path.join(processed_dir, 'mimic_cxr_processed.csv')
    if not os.path.exists(src):
        print(f'[MIMIC-CXR] skipped: {src} not found')
        return
    df = pd.read_csv(src, dtype=str).set_index('dicom_id')
    for split in ['train', 'val', 'test']:
        ids = pd.read_csv(os.path.join(splits_dir, 'mimic_cxr', f'{split}.csv'), dtype=str)
        missing = ~ids.dicom_id.isin(df.index)
        if missing.any():
            raise SystemExit(f'[MIMIC-CXR] {missing.sum()} {split} dicom_ids not found in {src}; '
                             'check that preprocessing used the same filters (frontal views, '
                             'non-empty findings).')
        out = df.loc[ids.dicom_id].reset_index()
        out.to_csv(os.path.join(processed_dir, f'mimic_cxr_{split}_clean.csv'), index=False)
        print(f'[MIMIC-CXR] {split}: {len(out)} images, {out.subject_id.nunique()} patients')


def rebuild_iu(processed_dir, splits_dir):
    src = os.path.join(processed_dir, 'iu_xray_all.csv')
    if not os.path.exists(src):
        print(f'[IU X-Ray] skipped: {src} not found')
        return
    df = pd.read_csv(src)
    df['filename'] = df.image_path.map(os.path.basename)
    df = df.drop_duplicates('filename').set_index('filename')
    for split in ['train', 'val', 'test']:
        ids = pd.read_csv(os.path.join(splits_dir, 'iu_xray', f'{split}.csv'))
        missing = ~ids.filename.isin(df.index)
        if missing.any():
            raise SystemExit(f'[IU X-Ray] {missing.sum()} {split} images not found in {src}')
        out = df.loc[ids.filename, ['image_path', 'findings']]
        out.to_csv(os.path.join(processed_dir, f'iu_xray_{split}_clean.csv'), index=False)
        print(f'[IU X-Ray] {split}: {len(out)} images')


def check_chexpert(processed_dir, splits_dir):
    src = os.path.join(processed_dir, 'chexpert_plus_test_clean.csv')
    if not os.path.exists(src):
        print(f'[CheXpert Plus] skipped: {src} not found')
        return
    df = pd.read_csv(src)
    rel = 'valid/' + df.image_path.str.split('png_valid/').str[1].str.replace('.png', '.jpg', regex=False)
    ids = pd.read_csv(os.path.join(splits_dir, 'chexpert_plus', 'test.csv'))
    if list(rel) != list(ids.path_to_image):
        raise SystemExit('[CheXpert Plus] preprocessed file does not match splits/chexpert_plus/test.csv')
    print(f'[CheXpert Plus] test: {len(df)} images (matches split file)')


def main():
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--processed_dir', default='./data/processed')
    parser.add_argument('--splits_dir', default=os.path.join(repo_root, 'splits'))
    args = parser.parse_args()

    rebuild_mimic(args.processed_dir, args.splits_dir)
    rebuild_iu(args.processed_dir, args.splits_dir)
    check_chexpert(args.processed_dir, args.splits_dir)


if __name__ == '__main__':
    main()
