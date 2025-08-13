# ASK1 IC50 Prediction - Advanced Jinwon

This module implements an advanced IC50 regression model for ASK1 inhibitors.
It combines chemoinformatics features with mechanism-aware flags derived from
four known ASK1 inhibition mechanisms:

1. **Thioredoxin binding suppression**
2. **14-3-3 binding via Ser967 phosphorylation**
3. **ATP-competitive inhibition through the hinge region**
4. **SCF Fbxo21 ubiquitin ligase modulation**

The pipeline now follows the data loading procedure from the baseline notebook
found in `step1_baseline/`. ChEMBL and PubChem data are loaded and merged as in
the notebook before standardization. Morgan fingerprints, MACCS keys,
physicochemical descriptors and heuristic mechanism flags are computed. Optionally, docking
scores (e.g. Gibbs free energy from AutoDock Vina or Schrödinger) can be added
when the required tools are installed. An XGBoost regressor trains on these
features and reports a validation RMSE. When `ensemble_size` in
`config/hyperparams.yaml` is greater than 1, multiple models are bootstrapped and
averaged to provide a mean prediction with a per-compound standard deviation.
After training, predictions for `data/test.csv` are saved to
`submissions/advanced_jinwon_submission.csv`.
A lightweight neural network alternative is available in `src/train/train_mlp_ensemble.py`.
It trains an ensemble of `MLPRegressor` models and writes predictions to `submission/DATE/submission_TIME.csv`.


## Usage

1. Install the dependencies defined in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the training script from the repository root. It can be invoked either
   directly or with the `-m` flag. The training pipeline now applies a simple
   SMOTE-based augmentation on the training set to mitigate overfitting.
   Hyperparameters are tuned via 5-fold cross validation and early
   stopping with a small validation split so training will halt automatically
   when the score stops improving:

   ```bash
   # Option A: execute directly
   python step2_advanced_jinwon/src/train/train_regression.py

   # Option B: run as a module
   python -m step2_advanced_jinwon.src.train.train_regression
   ```

After training, predictions for `data/test.csv` will be written to
`submissions/advanced_jinwon_submission.csv`.
To train an MLP ensemble instead, run `src/train/train_mlp_ensemble.py`. Results are placed in `submission/DATE/submission_TIME.csv`.


## Required Packages

The advanced training pipeline relies on the following libraries. All are
listed in `requirements.txt` and can be installed with `pip install -r requirements.txt`:

* pandas, numpy, scipy
* scikit-learn
* xgboost
* lightgbm
* optuna
* rdkit

## Running with SLURM

When a GPU cluster managed by SLURM is available, the helper script
`slurm_submit.py` can automatically submit the training job to the first free
GPU partition. The script cycles through `gpu1` to `gpu6`, waiting up to
10&nbsp;seconds in each partition for the job to receive a node allocation. If no
node is assigned after one full cycle, the job is left queued in `gpu1`.

```bash
python slurm_submit.py
```

The training script detects the `CUDA_VISIBLE_DEVICES` variable provided by
SLURM and will automatically switch XGBoost to GPU mode when a device is
available.
