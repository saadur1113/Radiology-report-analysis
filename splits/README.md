# Data splits

These files list the images in each split by identifier only. They contain no report text,
images or file paths. Use `src/data_prep/rebuild_splits.py` to turn them into full training
and evaluation tables from your own copies of the datasets (see the main README).

Rows are in the exact order used in the paper.

| File | Columns | Rows |
|---|---|---|
| `mimic_cxr/train.csv` | `dicom_id, study_id, subject_id` | 94,243 images, 41,455 patients |
| `mimic_cxr/val.csv` | `dicom_id, study_id, subject_id` | 11,822 images, 5,181 patients |
| `mimic_cxr/test.csv` | `dicom_id, study_id, subject_id` | 11,368 images, 5,183 patients |
| `iu_xray/{train,val,test}.csv` | `filename` (image file name in the Kaggle `images_normalized` release) | 2,239 / 320 / 640 |
| `chexpert_plus/test.csv` | `path_to_image` (as in `df_chexpert_plus_240401.csv`) | 72 |

**MIMIC-CXR.** These are frontal (PA/AP) images with a non-empty findings section. The splits
are patient-level, so no `subject_id` appears in more than one split. They were derived from
the official MIMIC-CXR-JPG split by moving patients that crossed split boundaries
(`src/data_prep/fix_patient_leakage.py`). Only the validation split is used for evaluation in
the paper. Access to MIMIC-CXR requires PhysioNet credentialing and acceptance of its data use
agreement.

**IU X-Ray.** Only the 640-image test split is used in the paper (zero-shot evaluation). The
train and validation lists are included for completeness.

**CheXpert Plus.** These are the 72 validation-set frontal images with a findings section,
used for zero-shot evaluation.
