# Does RAG Work Through Q-Former? A Controlled Analysis for Radiology Report Generation

Code for the WACV 2027 paper by **Saad Ur Rehman** and **Vikram Ravindra** (University of Cincinnati).

We add a second (retrieval) cross-attention layer to a BLIP-2 Q-Former and test whether it
uses retrieved radiology reports. We compare three retrieval strategies plus an oracle query,
six decoder configurations, and three hospital systems (MIMIC-CXR, IU X-Ray, CheXpert Plus).
Under standard training the retrieval pathway is starved of gradient and the model bypasses it.
Forced-utilization training shows that the pathway itself can route content.

Paper: [`paper/RAG_QFormer_WACV2027.pdf`](paper/RAG_QFormer_WACV2027.pdf)

This repository contains the code, the paper and the patient-level data splits. It does not contain
data, model checkpoints, or predictions (see [Data](#data)).

## Repository layout

```
paper/                      Camera-ready paper (PDF)
splits/                     ID-only train/val/test splits (see splits/README.md)
src/
  model/                    Dual cross-attention Q-Former + OPT, datasets
  data_prep/                Dataset preprocessing, patient-level split construction, split rebuild
  retrieval/                Stage 1 drafts, FAISS indices and retrieval caches
  training/                 Stage 1 / Stage 2 training and all training ablations
  evaluation/               Generation metrics, RadGraph F1, length control, content ablation,
                            significance tests, VLM baselines, qualitative examples
```

All scripts are run from the repository root. Each script's docstring and `--help` list its
full set of arguments.

## Setup

Experiments were run with Python 3.9, PyTorch 2.6 (CUDA 12.4) and Transformers 4.56 on
single NVIDIA GPUs.

```bash
conda create -n qformer-rag python=3.9 && conda activate qformer-rag
pip install -r requirements.txt
conda install -c pytorch faiss-gpu   # or: pip install faiss-cpu
```

The pretrained weights download from the Hugging Face Hub on first use:
`Salesforce/blip2-opt-2.7b`, `Salesforce/blip2-opt-6.7b`,
`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` (frozen retrieval-report encoder inside
the Q-Former), `pritamdeka/S-PubMedBert-MS-MARCO` (FAISS retrieval encoder), BiomedCLIP
(`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) and `Qwen/Qwen2-VL-7B-Instruct`.

## Data

None of the datasets can be redistributed here. Obtain them from their original sources
under their own terms:

| Dataset | Source | Used for |
|---|---|---|
| MIMIC-CXR-JPG v2 + MIMIC-CXR reports | [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/) (credentialed access, DUA) | training, retrieval corpus, main evaluation |
| IU X-Ray (Open-i) | [Open-i / NLM](https://openi.nlm.nih.gov/) | zero-shot test (640 images) |
| CheXpert Plus | [Stanford AIMI](https://aimi.stanford.edu/datasets/chexpert-plus) | zero-shot test (72 validation images) |

### Rebuilding the exact splits

`splits/` lists the images in each split by identifier only (no report text or file paths).
Splits are patient-level: no MIMIC-CXR patient appears in more than one split.

| Split | MIMIC-CXR images (patients) | IU X-Ray | CheXpert Plus |
|---|---|---|---|
| train | 94,243 (41,455) | 2,239 | – |
| val | 11,822 (5,181) | 320 | – |
| test | 11,368 (5,183) | 640 | 72 |

```bash
# 1. Per-image tables from your local copies of the datasets
python src/data_prep/prepare_mimic_data_separate_folders.py \
    --image_dir /path/to/mimic-cxr-jpg/files --report_dir /path/to/mimic-cxr-reports/files \
    --metadata_csv /path/to/mimic-cxr-2.0.0-metadata.csv.gz \
    --split_csv /path/to/mimic-cxr-2.0.0-split.csv.gz \
    --filter_frontal --output_dir ./data/processed
python src/data_prep/preprocess_iu_xray.py --data_dir /path/to/iu_xray --output_dir ./data/processed
CHEXPERT_PLUS_DIR=/path/to/chexpert_plus python src/data_prep/preprocess_chexpert_plus.py

# 2. Select the paper's rows, in the paper's order
python src/data_prep/rebuild_splits.py --processed_dir ./data/processed
```

This writes `data/processed/mimic_cxr_{train,val,test}_clean.csv` and
`iu_xray_{train,val,test}_clean.csv`. On our data the rebuilt files are byte-identical to the
ones used in the paper. Row order matters: the VLM baselines (Table 1) and the matched-decoding
comparison (Table 5) use the first 1,000 rows of `mimic_cxr_val_clean.csv`.

The patient-level splits were originally made with `fix_patient_leakage.py` (it reassigns
patients that crossed the official MIMIC-CXR split boundaries). `check_data_leakage.py`,
`check_data_leakage_advanced.py` and `verify_split_logic.py` audit the result.

> **Note:** several training scripts default to `mimic_cxr_{train,val}.csv`, the splits from
> before the patient-level fix. Always pass the `*_clean.csv` files explicitly, as in the
> commands below.

## Reproducing the paper

The commands below use the following shorthands:

```bash
TRAIN=./data/processed/mimic_cxr_train_clean.csv
VAL=./data/processed/mimic_cxr_val_clean.csv
```

### 1. Stage 1: image-to-report (no retrieval)

```bash
python src/training/train_stage1_opt.py --train_csv $TRAIN --val_csv $VAL \
    --output_dir ./checkpoints/stage1 --seed 42        # Table 1 multi-seed: 42, 123, 456
```

### 2. Generate-then-retrieve cache (Stage 2 training context)

Stage 1 drafts are used as text queries against a FAISS index of S-PubMedBert-MS-MARCO
embeddings of the training reports (k = 3).

```bash
python src/retrieval/generate_stage1_drafts.py --checkpoint ./checkpoints/stage1/best_model.pt \
    --csv_file $TRAIN --output_file ./data/gtr/stage1_drafts_train.pkl
python src/retrieval/generate_stage1_drafts.py --checkpoint ./checkpoints/stage1/best_model.pt \
    --csv_file $VAL --output_file ./data/gtr/stage1_drafts_val.pkl
#   (generate_remaining_drafts.py resumes an interrupted draft run from a given index)
python src/retrieval/build_gen_then_retrieve_cache.py --train_csv $TRAIN --val_csv $VAL \
    --train_drafts ./data/gtr/stage1_drafts_train.pkl --val_drafts ./data/gtr/stage1_drafts_val.pkl \
    --output_dir ./data/gtr
```

### 3. Stage 2: RAG-enhanced training

```bash
python src/training/train_stage2_opt_fast.py --stage1_checkpoint ./checkpoints/stage1/best_model.pt \
    --train_csv $TRAIN --val_csv $VAL \
    --train_cache ./data/gtr/retrieval_cache_train_gtr.pkl --val_cache ./data/gtr/retrieval_cache_val_gtr.pkl \
    --output_dir ./checkpoints/stage2
```

### 4. Table 1: main MIMIC-CXR results

```bash
S1=./checkpoints/stage1/best_model.pt; S2=./checkpoints/stage2/best_model.pt

# Stage 1 (no RAG)
python src/evaluation/evaluate_models_comprehensive_fast.py --checkpoint $S1 --csv_file $VAL --output_dir ./results/stage1

# + ViT image retrieval
python src/retrieval/build_image_retrieval_index.py --stage1_model $S1 --train_csv $TRAIN --output_dir ./data/image_retrieval
python src/evaluation/evaluate_image_retrieval_FIXED.py --checkpoint $S2 --csv_file $VAL \
    --image_index_dir ./data/image_retrieval --output_dir ./results/vit_retrieval

# + Generate-then-retrieve
python src/evaluation/evaluate_gen_then_retrieve.py --checkpoint $S2 --csv_file $VAL \
    --val_cache ./data/gtr/retrieval_cache_val_gtr.pkl --output_dir ./results/gen_then_retrieve

# + BiomedCLIP retrieval
python src/evaluation/evaluate_biomedclip_retrieval.py --mode build_index --train_csv $TRAIN --index_dir ./data/biomedclip_retrieval
python src/evaluation/evaluate_biomedclip_retrieval.py --mode evaluate --checkpoint $S2 --csv_file $VAL \
    --index_dir ./data/biomedclip_retrieval --output_dir ./results/biomedclip --batch_size 8

# + Oracle query (ground-truth findings used as the retrieval query)
python src/retrieval/make_oracle_queries.py --csv_file $VAL --output_file ./data/oracle/oracle_val_drafts.pkl
python src/retrieval/build_gen_then_retrieve_cache.py --train_csv $TRAIN --val_csv $VAL \
    --train_drafts ./data/gtr/stage1_drafts_train.pkl --val_drafts ./data/oracle/oracle_val_drafts.pkl \
    --output_dir ./data/oracle
python src/evaluation/evaluate_gen_then_retrieve.py --checkpoint $S2 --csv_file $VAL \
    --val_cache ./data/oracle/retrieval_cache_val_gtr.pkl --output_dir ./results/oracle

# RadGraph F1 for any run
python src/evaluation/evaluate_radgraph_final.py --predictions_file ./results/<run>/predictions.json --output_dir ./results/<run>

# Qwen2-VL baselines (1,000-sample subset) and the shared 1,000-sample comparison
python src/evaluation/evaluate_qwen2vl_baseline.py --csv_file $VAL --max_samples 1000 --output_dir ./results/qwen2vl_baseline
python src/evaluation/evaluate_qwen2vl_prompt_eng.py --csv_file $VAL --max_samples 1000 --output_dir ./results/qwen2vl_prompt_eng
python src/evaluation/consistent_sample_eval.py --csv $VAL --output_dir ./results/consistent_1000

# Paired Wilcoxon signed-rank / bootstrap tests
python src/evaluation/statistical_tests.py --csv $VAL --output_dir ./results/significance_tests
```

`compute_metrics_from_csv.py` recomputes metrics for a predictions file that does not contain
references. `consistent_sample_eval.py` and `statistical_tests.py` read predictions from fixed
paths under `./results/`; edit the paths at the top of each script to match your run names.

### 5. Table 2: cross-institutional zero-shot evaluation

Run the Table 1 evaluation scripts with
`--csv_file ./data/processed/iu_xray_test_clean.csv` or
`--csv_file ./data/processed/chexpert_plus_test_clean.csv`. Keep the MIMIC-CXR training indices
as the retrieval source.

### 6. Table 3: length-controlled evaluation

```bash
for W in 35 39; do
  python src/evaluation/truncate_and_score.py --predictions ./results/<run>/predictions.json \
      --csv $VAL --target_length $W --output_dir ./results/<run>_trunc$W
done
```

IU X-Ray uses `--target_length` 25, 31 and 38.

### 7. Table 4: decoder and Q-Former ablations

Each run below takes the same Stage 2 arguments as step 3 (`--stage1_checkpoint`,
`--train_csv $TRAIN --val_csv $VAL`, `--train_cache`, `--val_cache`, `--output_dir`).

| Row | Script |
|---|---|
| OPT-2.7B unfrozen | `src/training/train_stage2_unfrozen.py` |
| + Higher retrieval LR | `src/training/train_stage2_fix1_higher_lr.py` |
| + Pre-initialized retrieval | `src/training/train_stage2_fix2_preinit_v2.py` |
| + Both combined | `src/training/train_stage2_fix3_combined_v2.py` |
| OPT-6.7B frozen | `src/training/train_stage1_opt_6.7b.py`, then `src/training/train_stage2_opt_6.7b.py` |

Evaluate each run without retrieval (`evaluate_models_comprehensive_fast.py`) and with
BiomedCLIP retrieval (`evaluate_biomedclip_retrieval.py --mode evaluate`). For OPT-6.7B, use
`evaluate_comprehensive_6.7b.py` and `evaluate_biomedclip_6.7b.py`.

### 8. Table 5: direct decoder integration and matched decoding

```bash
python src/training/train_direct_context.py --stage1_checkpoint $S1 \
    --retrieval_cache_train ./data/gtr/retrieval_cache_train_gtr.pkl \
    --retrieval_cache_val ./data/gtr/retrieval_cache_val_gtr.pkl --output_dir ./checkpoints/direct_context
python src/training/train_direct_context_nocontext.py --stage1_checkpoint $S1 \
    --retrieval_cache_train ./data/gtr/retrieval_cache_train_gtr.pkl \
    --retrieval_cache_val ./data/gtr/retrieval_cache_val_gtr.pkl --output_dir ./checkpoints/direct_context_nocontext
python src/evaluation/eval_direct_vs_qformer.py --models stage1,qformer,direct,control \
    --stage1_ckpt $S1 --qformer_ckpt $S2 \
    --direct_ckpt ./checkpoints/direct_context/best_model.pt \
    --control_ckpt ./checkpoints/direct_context_nocontext/best_model.pt \
    --val_cache ./data/gtr/retrieval_cache_val_gtr.pkl --output_dir ./results/direct_vs_qformer
```

`eval_direct_vs_qformer.py` also runs the retrieval content ablation (pangram and
swapped-context tests) and the bootstrap confidence intervals.

### 9. Table 6: diagnostic experiments

| Exp | Stage 1 | Stage 2 | Evaluation |
|---|---|---|---|
| A: remove residual | `train_stage1_opt.py` | `train_stage2_no_residual.py` | `evaluate_no_rag_no_residual.py`, `evaluate_biomedclip_no_residual.py` |
| B: random retrieval in Stage 1 | `train_stage1_with_retrieval.py --seed {42,123,456}` | `train_stage2_opt_fast.py` | standard |
| B′: relevant retrieval in Stage 1 | `train_stage1_relevant_retrieval.py --retrieval_cache <cache>` | `train_stage2_opt_fast.py` | standard |
| C: A + B | `train_stage1_with_retrieval.py` | `train_stage2_no_residual.py` | as A |
| D: forced utilization | `train_stage1_opt.py` | `train_stage2_forced_utilization.py --mask_prob 1.0 --seed {0,123,456}` | `evaluate_expD.py` |

The Exp B′ cache holds BiomedCLIP-retrieved reports for every training image:

```bash
python src/retrieval/build_stage1_biomedclip_cache.py --train_csv $TRAIN \
    --biomedclip_index_dir ./data/biomedclip_retrieval --output_path ./data/stage1_biomedclip_cache/retrieval_cache_train.pkl
```

`evaluate_expD.py` runs the blank-image swapped-context and pangram tests (100 samples). It
can be applied to any Stage 2 checkpoint, including the standard model.

### 10. Gradient and weight analysis (Section 3.6)

```bash
python src/training/train_stage2_gradient_logging.py --stage1_checkpoint $S1 \
    --train_csv $TRAIN --val_csv $VAL \
    --train_cache ./data/gtr/retrieval_cache_train_gtr.pkl --val_cache ./data/gtr/retrieval_cache_val_gtr.pkl \
    --output_dir ./checkpoints/stage2_gradient_logging
```

This trains standard Stage 2 (real images, generate-then-retrieve context) and writes
`gradient_norms.log` (every 50 steps) and `weight_stats.log` (per epoch).
`train_stage2_forced_utilization.py` logs the same statistics for Exp D.

### 11. Figure 2: qualitative examples

`src/evaluation/find_best_qualitative_examples.py` searches Stage 1, BiomedCLIP and Qwen2-VL
predictions for the four case types shown in Figure 2. Edit the prediction paths at the top of
the script.

## Citation

```bibtex
@inproceedings{rehman2027ragqformer,
  title     = {Does {RAG} Work Through {Q-Former}? A Controlled Analysis for Radiology Report Generation},
  author    = {Ur Rehman, Saad and Ravindra, Vikram},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## License

Code is released under the MIT License (see `LICENSE`). The datasets keep their own licenses
and data use agreements.
