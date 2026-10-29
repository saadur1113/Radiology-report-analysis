"""
MIMIC-CXR Data Preparation Script
For setup with separate image and report folders

This script:
1. Matches JPG images to TXT reports using metadata
2. Parses report sections (FINDINGS, IMPRESSION)
3. Creates train/val/test splits
4. Prepares data for RAG-enhanced BLIP2 training
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import logging
from typing import Dict, List, Tuple


def setup_logging(output_dir: str):
    """Setup logging to both file and console"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'data_preparation.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare MIMIC-CXR data with separate image/report folders'
    )
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing JPG images (e.g., /path/to/images)')
    parser.add_argument('--report_dir', type=str, required=True,
                        help='Directory containing TXT reports (e.g., /path/to/reports)')
    parser.add_argument('--metadata_csv', type=str, required=True,
                        help='Path to metadata CSV (e.g., mimic-cxr-2.0.0-metadata.csv)')
    parser.add_argument('--split_csv', type=str, default=None,
                        help='Path to split CSV (optional)')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Sample size for testing (None for full dataset)')
    parser.add_argument('--filter_frontal', action='store_true',
                        help='Keep only frontal (PA/AP) images')
    parser.add_argument('--validate_files', action='store_true',
                        help='Validate that image and report files exist')
    parser.add_argument('--min_finding_length', type=int, default=10,
                        help='Minimum findings text length')
    parser.add_argument('--min_impression_length', type=int, default=5,
                        help='Minimum impression text length')
    return parser.parse_args()


def load_metadata(metadata_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Load MIMIC-CXR metadata"""
    logger.info(f"Loading metadata from: {metadata_path}")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata = pd.read_csv(metadata_path)
    logger.info(f"✓ Loaded metadata: {len(metadata)} records")
    logger.info(f"  Columns: {list(metadata.columns)}")
    
    # Check required columns
    required = ['subject_id', 'study_id', 'dicom_id']
    missing = [col for col in required if col not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")
    
    logger.info(f"  subject_id range: {metadata['subject_id'].min():,} to {metadata['subject_id'].max():,}")
    logger.info(f"  study_id range:   {metadata['study_id'].min():,} to {metadata['study_id'].max():,}")
    
    return metadata


def construct_image_path(row, image_dir: str) -> str:
    """
    Construct path to JPG image file
    
    Expected structure:
    image_dir/p{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg
    """
    subject_id = str(row['subject_id'])
    study_id = str(row['study_id'])
    dicom_id = str(row['dicom_id'])
    
    # Get first 2 digits of subject_id for subdirectory
    subject_prefix = subject_id[:2] if len(subject_id) >= 2 else '10'
    
    image_path = os.path.join(
        image_dir,
        f'p{subject_prefix}',
        f'p{subject_id}',
        f's{study_id}',
        f'{dicom_id}.jpg'
    )
    
    return image_path


def construct_report_path(row, report_dir: str) -> str:
    """
    Construct path to TXT report file
    
    Expected structure:
    report_dir/p{prefix}/p{subject_id}/s{study_id}.txt
    """
    subject_id = str(row['subject_id'])
    study_id = str(row['study_id'])
    
    # Get first 2 digits of subject_id for subdirectory
    subject_prefix = subject_id[:2] if len(subject_id) >= 2 else '10'
    
    report_path = os.path.join(
        report_dir,
        f'p{subject_prefix}',
        f'p{subject_id}',
        f's{study_id}.txt'
    )
    
    return report_path


def parse_report_sections(report_text: str) -> Dict[str, str]:
    """
    Parse radiology report into structured sections
    
    Returns dict with keys: findings, impression, indication, etc.
    """
    sections = {}
    section_headers = ['INDICATION', 'FINDINGS', 'IMPRESSION', 'TECHNIQUE', 'COMPARISON']
    
    lines = report_text.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        
        # Check if this line is a section header
        header_found = False
        for header in section_headers:
            if line.startswith(header + ':') or line == header:
                # Save previous section
                if current_section:
                    sections[current_section.lower()] = ' '.join(current_content).strip()
                
                # Start new section
                current_section = header
                content_start = line.replace(header + ':', '').replace(header, '').strip()
                current_content = [content_start] if content_start else []
                header_found = True
                break
        
        # If not a header and we're in a section, add to content
        if not header_found and current_section and line:
            current_content.append(line)
    
    # Save last section
    if current_section:
        sections[current_section.lower()] = ' '.join(current_content).strip()
    
    return sections


def load_and_parse_report(report_path: str) -> Dict[str, str]:
    """Load and parse a single report file"""
    try:
        with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
            report_text = f.read()
        
        sections = parse_report_sections(report_text)
        sections['full_text'] = report_text
        return sections
    
    except Exception as e:
        return {
            'findings': '',
            'impression': '',
            'indication': '',
            'full_text': '',
            'error': str(e)
        }


def process_dataset(
    metadata: pd.DataFrame,
    image_dir: str,
    report_dir: str,
    logger: logging.Logger,
    filter_frontal: bool = False,
    validate_files: bool = False,
    sample_size: int = None
) -> pd.DataFrame:
    """
    Main processing function
    
    Links images to reports and parses report content
    """
    
    logger.info("\n" + "=" * 70)
    logger.info("PROCESSING DATASET")
    logger.info("=" * 70)
    
    # Filter to frontal views if requested
    if filter_frontal and 'ViewPosition' in metadata.columns:
        original_len = len(metadata)
        metadata = metadata[metadata['ViewPosition'].isin(['PA', 'AP'])].copy()
        logger.info(f"✓ Filtered to frontal views: {len(metadata)} (removed {original_len - len(metadata)})")
    
    # Sample if requested
    if sample_size and sample_size < len(metadata):
        logger.info(f"Sampling {sample_size} records for testing...")
        metadata = metadata.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    # Construct file paths
    logger.info("\nConstructing file paths...")
    metadata['image_path'] = metadata.apply(
        lambda row: construct_image_path(row, image_dir), 
        axis=1
    )
    metadata['report_path'] = metadata.apply(
        lambda row: construct_report_path(row, report_dir), 
        axis=1
    )
    
    # Validate files exist if requested
    if validate_files:
        logger.info("\nValidating file existence...")
        
        metadata['image_exists'] = metadata['image_path'].apply(os.path.exists)
        metadata['report_exists'] = metadata['report_path'].apply(os.path.exists)
        
        img_exists = metadata['image_exists'].sum()
        rep_exists = metadata['report_exists'].sum()
        
        logger.info(f"  Images found:  {img_exists}/{len(metadata)} ({img_exists/len(metadata)*100:.1f}%)")
        logger.info(f"  Reports found: {rep_exists}/{len(metadata)} ({rep_exists/len(metadata)*100:.1f}%)")
        
        # Keep only records where both exist
        metadata = metadata[metadata['image_exists'] & metadata['report_exists']].copy()
        metadata = metadata.drop(columns=['image_exists', 'report_exists'])
        
        logger.info(f"  ✓ Keeping {len(metadata)} records with both files")
    
    # Parse reports
    logger.info("\nParsing report files...")
    reports_data = []
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Loading reports"):
        report_sections = load_and_parse_report(row['report_path'])
        reports_data.append(report_sections)
    
    # Add report sections to metadata
    reports_df = pd.DataFrame(reports_data)
    result_df = pd.concat([metadata.reset_index(drop=True), reports_df], axis=1)
    
    # Create combined text for RAG
    logger.info("\nCreating combined text for RAG embeddings...")
    result_df['combined_text'] = result_df.apply(
        lambda row: f"FINDINGS: {row['findings']}\n\nIMPRESSION: {row['impression']}",
        axis=1
    )
    
    logger.info(f"✓ Processed {len(result_df)} image-report pairs")
    
    return result_df


def filter_quality(
    data_df: pd.DataFrame,
    logger: logging.Logger,
    min_findings_length: int = 10,
    min_impression_length: int = 5
) -> pd.DataFrame:
    """Filter out low-quality reports"""
    
    logger.info("\nFiltering for quality...")
    original_len = len(data_df)
    
    # Fill NaN
    data_df['findings'] = data_df['findings'].fillna('')
    data_df['impression'] = data_df['impression'].fillna('')
    
    # Filter by length
    data_df = data_df[
        (data_df['findings'].str.len() >= min_findings_length) &
        (data_df['impression'].str.len() >= min_impression_length)
    ].copy()
    
    removed = original_len - len(data_df)
    logger.info(f"✓ Removed {removed} low-quality entries ({removed/original_len*100:.1f}%)")
    logger.info(f"✓ Remaining: {len(data_df)} high-quality entries")
    
    return data_df


def create_splits(
    data_df: pd.DataFrame,
    logger: logging.Logger,
    split_csv: str = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> pd.DataFrame:
    """Create or load train/val/test splits"""
    
    logger.info("\nCreating train/val/test splits...")
    
    if split_csv and os.path.exists(split_csv):
        # Use official MIMIC splits
        logger.info(f"Loading official splits from: {split_csv}")
        splits = pd.read_csv(split_csv)
        
        # Merge with data
        data_df = data_df.merge(
            splits[['dicom_id', 'split']],
            on='dicom_id',
            how='left'
        )
        
        # Fill any missing splits with 'train'
        data_df['split'] = data_df['split'].fillna('train')
    
    else:
        # Create random splits
        logger.info("Creating random splits...")
        data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n = len(data_df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        data_df['split'] = 'test'
        data_df.loc[:train_end, 'split'] = 'train'
        data_df.loc[train_end:val_end, 'split'] = 'val'
    
    # Report split sizes
    train_count = sum(data_df['split'] == 'train')
    val_count = sum(data_df['split'] == 'val')
    test_count = sum(data_df['split'] == 'test')
    
    logger.info(f"✓ Train: {train_count} ({train_count/len(data_df)*100:.1f}%)")
    logger.info(f"✓ Val:   {val_count} ({val_count/len(data_df)*100:.1f}%)")
    logger.info(f"✓ Test:  {test_count} ({test_count/len(data_df)*100:.1f}%)")
    
    return data_df


def save_processed_data(data_df: pd.DataFrame, output_dir: str, logger: logging.Logger):
    """Save processed data to CSV files"""
    
    logger.info("\nSaving processed data...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full dataset
    full_path = os.path.join(output_dir, 'mimic_cxr_processed.csv')
    data_df.to_csv(full_path, index=False)
    logger.info(f"✓ Saved: {full_path}")
    
    # Save individual splits
    for split in ['train', 'val', 'test']:
        split_df = data_df[data_df['split'] == split]
        split_path = os.path.join(output_dir, f'mimic_cxr_{split}.csv')
        split_df.to_csv(split_path, index=False)
        logger.info(f"✓ Saved: {split_path} ({len(split_df)} samples)")
    
    # Save dataset statistics
    stats = {
        'total_samples': int(len(data_df)),
        'train_samples': int(len(data_df[data_df['split'] == 'train'])),
        'val_samples': int(len(data_df[data_df['split'] == 'val'])),
        'test_samples': int(len(data_df[data_df['split'] == 'test'])),
        'avg_findings_length': float(data_df['findings'].str.len().mean()),
        'avg_impression_length': float(data_df['impression'].str.len().mean()),
        'avg_combined_text_length': float(data_df['combined_text'].str.len().mean()),
        'columns': list(data_df.columns),
    }
    
    if 'ViewPosition' in data_df.columns:
        stats['view_position_distribution'] = data_df['ViewPosition'].value_counts().to_dict()
    
    stats_path = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✓ Saved: {stats_path}")
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total samples: {stats['total_samples']:,}")
    logger.info(f"Train: {stats['train_samples']:,}")
    logger.info(f"Val:   {stats['val_samples']:,}")
    logger.info(f"Test:  {stats['test_samples']:,}")
    logger.info(f"\nAverage text lengths:")
    logger.info(f"  Findings:   {stats['avg_findings_length']:.1f} chars")
    logger.info(f"  Impression: {stats['avg_impression_length']:.1f} chars")
    logger.info(f"  Combined:   {stats['avg_combined_text_length']:.1f} chars")
    
    return output_dir


def main():
    args = parse_args()
    logger = setup_logging(args.output_dir)
    
    logger.info("=" * 70)
    logger.info("MIMIC-CXR DATA PREPARATION")
    logger.info("Separate Image and Report Folders Configuration")
    logger.info("=" * 70)
    logger.info(f"Image directory:  {args.image_dir}")
    logger.info(f"Report directory: {args.report_dir}")
    logger.info(f"Metadata CSV:     {args.metadata_csv}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Load metadata
        metadata = load_metadata(args.metadata_csv, logger)
        
        # Process dataset
        data_df = process_dataset(
            metadata,
            args.image_dir,
            args.report_dir,
            logger,
            filter_frontal=args.filter_frontal,
            validate_files=args.validate_files,
            sample_size=args.sample_size
        )
        
        # Filter quality
        data_df = filter_quality(
            data_df,
            logger,
            min_findings_length=args.min_finding_length,
            min_impression_length=args.min_impression_length
        )
        
        # Create splits
        data_df = create_splits(
            data_df,
            logger,
            split_csv=args.split_csv
        )
        
        # Save
        output_dir = save_processed_data(data_df, args.output_dir, logger)
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ DATA PREPARATION COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"\nProcessed data saved to: {output_dir}")
        logger.info("\nNext steps:")
        logger.info("1. Build retrieval index (build_retrieval_index.py)")
        logger.info("2. Create dataset class for training")
        logger.info("3. Train RAG-enhanced Q-Former")
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()