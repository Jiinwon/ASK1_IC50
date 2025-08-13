# ASK1 IC50 Prediction - Advanced Jinwon

This module implements an advanced IC50 regression model for ASK1 inhibitors.
It combines chemoinformatics features with mechanism-aware flags derived from
four known ASK1 inhibition mechanisms:

1. **Thioredoxin binding suppression**
2. **14-3-3 binding via Ser967 phosphorylation**
3. **ATP-competitive inhibition through the hinge region**
4. **SCF Fbxo21 ubiquitin ligase modulation**

The training script now builds the dataset directly from the raw CAS,
ChEMBL and PubChem files located in `data/`. For each source the SMILES
strings and pIC50 values are extracted and merged into a single
`raw_input.csv`. Canonical SMILES are generated, duplicates are removed
and the result is saved as `processed_input.csv`. The data is then split
into `processed_train.csv` and `processed_val.csv` for model training.
Histograms of all four datasets and of the test-set predictions are
written to `results/`.

Morgan fingerprints, MACCS keys, physicochemical descriptors and the
mechanism flags are computed on the processed training data. Optionally,
docking scores and complex-structure descriptors can be appended when
the required tools are installed. An XGBoost regressor trains on these
features and reports a validation RMSE. When `use_complex_features` is
set in `config/hyperparams.yaml`, simple descriptors summarising ASK1
protein–protein complex interfaces are included. After training,
predictions for `data/test.csv` are written to the dated subdirectory
under `submissions/`.
A lightweight neural network alternative is available in `src/train/train_mlp_ensemble.py`.
It trains an ensemble of `MLPRegressor` models and writes predictions to `submission/DATE/submission_TIME.csv`.


## Usage

1. Install the dependencies defined in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the training script from the repository root. It can be invoked either
   directly or with the `-m` flag. The script automatically loads the raw CAS,
   ChEMBL and PubChem datasets, imputes missing IC50 values and builds
   feature caches before training. Hyperparameters are tuned via 5-fold
   cross validation and early stopping with a small validation split so
   training halts automatically when the score stops improving:

   ```bash
   # Option A: execute directly
   python step2_advanced_jinwon/src/train/train_regression.py

   # Option B: run as a module
   python -m step2_advanced_jinwon.src.train.train_regression
   ```

After training, predictions for `data/test.csv` will be written to
`submissions/advanced_jinwon_submission.csv`.
To train an MLP ensemble instead, run `src/train/train_mlp_ensemble.py`. Results are placed in `submission/DATE/submission_TIME.csv`.

To activate the complex-structure features, set `use_complex_features: true`
in `config/hyperparams.yaml` and ensure `complex_structures_dir` in
`config/paths.yaml` points to the directory containing the ASK1 complex PDB
files.


## Required Packages

The advanced training pipeline relies on the following libraries. All are
listed in `requirements.txt` and can be installed with `pip install -r requirements.txt`:

* pandas, numpy, scipy
* scikit-learn
* xgboost
* lightgbm
* optuna
* rdkit
* biopython

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
