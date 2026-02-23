# AI Cyber Lab 2 — Phishing URL Detection (Project Guide)

This document summarizes the full project after reviewing all repository files.

## 1) What this project does

This repository builds a **binary phishing URL classifier** using the **PhiUSIIL dataset** and a baseline **Random Forest** model.

Core workflow:
1. Load raw dataset CSV
2. Clean data (duplicates, optional missing-value handling, drop helper columns)
3. Auto-detect and normalize target labels
4. Train/test split with stratification
5. Standardize features
6. Train and persist model artifacts
7. Evaluate model and generate metrics + confusion matrix plot
8. Explore the dataset visually in a Jupyter notebook

---

## 2) Repository structure

- `src/data.py` — data loading + preprocessing pipeline
- `src/train.py` — model creation/training/saving pipeline
- `src/eval.py` — evaluation pipeline and confusion matrix generation
- `src/utils.py` — small utility helpers for directories/config
- `src/__init__.py` — package metadata
- `explore_dataset.py` — quick dataset inspection script
- `notebooks/01_eda.ipynb` — full EDA notebook with plots + insights
- `results/*.png` — generated visual artifacts from EDA/model evaluation
- `requirements.txt` — pinned dependencies
- `README.md` — minimal setup instructions

---

## 3) Environment setup

```bash
python -m pip install -r requirements.txt
```

Dependencies include:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyter
- joblib

---

## 4) Data expectations

The project expects this raw file path by default:

```text
data/raw/PhiUSIIL_Phishing_URL_Dataset.csv
```

If the CSV is missing, data loading raises `FileNotFoundError`.

Target label handling in `prepare_features()`:
- Auto-detects common target names (`label`, `target`, `phishing`, `CLASS_LABEL`, `class`, `Label`)
- Falls back to the **last column** if no known name matches
- Normalizes binary labels to `{0,1}` when needed

---

## 5) How to run

### A) Quick dataset inspection
```bash
python explore_dataset.py
```
Prints shape, columns, sample rows, dtypes, candidate target distribution, missing values, and statistical summary.

### B) Train model
```bash
python -m src.train
```
Outputs:
- `results/model.joblib` containing:
  - trained model
  - fitted scaler

### C) Evaluate model
```bash
python -m src.eval
```
Outputs:
- `results/metrics.json`
- `results/confusion_matrix.png`
- Console classification report (precision/recall/F1 by class)

---

## 6) Implementation notes by module

### `src/data.py`
- `load_dataset()` reads CSV and reports row/column count
- `clean_data()`:
  - drops duplicate rows
  - removes `FILENAME` column if present
  - drops columns with >50% missing values
  - fills remaining numeric missing values with median
- `prepare_features()`:
  - target detection + binary normalization
  - drops non-numeric feature columns
- `split_and_scale()`:
  - stratified split (`test_size=0.2`, `random_state=42`)
  - `StandardScaler` on features
- `get_processed_data()` composes full pipeline and returns
  `(X_train, X_test, y_train, y_test, scaler)`

### `src/train.py`
- Baseline model is `RandomForestClassifier` with:
  - `n_estimators=100`
  - `max_depth=10`
  - `random_state=42`
  - `n_jobs=-1`
- Trains on processed data and saves artifacts using `joblib`

### `src/eval.py`
- Loads `results/model.joblib`
- Rebuilds processed train/test split using the same data pipeline
- Computes:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Saves metrics JSON and confusion matrix image
- Prints a detailed classification report

### `src/utils.py`
- `ensure_dir()`
- `load_config()`
- `save_config()`

---

## 7) Notebook + visual outputs

`notebooks/01_eda.ipynb` contains a stepwise EDA flow:
- dataset loading/overview
- quality checks (missing + duplicates)
- target/class analysis
- correlation exploration
- feature-level distribution analysis
- written insights and conclusions

Current `results/` images include:
- `class_distribution.png`
- `correlation_matrix.png`
- `feature_importance.png`
- `features_by_class.png`
- `top_feature_distributions.png`
- `confusion_matrix.png`

---

## 8) Typical workflow for contributors

1. Place raw CSV at `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv`
2. Install deps from `requirements.txt`
3. Run EDA notebook/script to understand distributions
4. Train with `python -m src.train`
5. Evaluate with `python -m src.eval`
6. Compare metrics/plots before modifying model choices

---

## 9) Improvement ideas

- Add CLI arguments for data path, model hyperparameters, sample size
- Persist exact train/test split or seeds for strict reproducibility
- Add feature importance export to CSV
- Add unit tests for data preprocessing functions
- Add model registry/versioning metadata in output artifacts
- Expand to compare multiple baseline models (LogReg, XGBoost, SVM)
