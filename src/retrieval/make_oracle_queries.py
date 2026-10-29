"""
Oracle-query condition (Table 1, "+ Oracle query"): use each validation
sample's ground-truth findings as the retrieval *query*, in place of the
Stage 1 draft. The output has the same format as the drafts written by
generate_stage1_drafts.py ({row_index: text}), so it can be passed to
build_gen_then_retrieve_cache.py as --val_drafts.

Usage:
    python src/retrieval/make_oracle_queries.py \
        --csv_file ./data/processed/mimic_cxr_val_clean.csv \
        --output_file ./data/oracle/oracle_val_drafts.pkl
"""
import argparse
import os
import pickle

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', default='./data/processed/mimic_cxr_val_clean.csv')
    parser.add_argument('--output_file', default='./data/oracle/oracle_val_drafts.pkl')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    queries = {i: str(text) for i, text in enumerate(df['findings'])}
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(queries, f)
    print(f'Saved {len(queries)} oracle queries to {args.output_file}')


if __name__ == '__main__':
    main()
