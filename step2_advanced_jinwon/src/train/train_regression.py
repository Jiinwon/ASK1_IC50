"""Simplified training script implementing scaffold CV and competition metrics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure project root is on ``sys.path`` so imports like ``src`` and ``utils``
# work when the script is executed directly (``python train_regression.py``).
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Optional, Callable

import math
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.stats
from scipy.stats import gaussian_kde
from datetime import datetime
from zoneinfo import ZoneInfo
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
try:
    from imblearn.over_sampling import SMOTER, SMOGN
except Exception:  # pragma: no cover - imbalanced-learn not installed
    SMOTER = SMOGN = None  # type: ignore

from src.data.impute import build_feature_pipeline
from src.data.splitters import compute_scaffolds
from src.data.standardize_smiles import apply_standardization
from src.featurization.ecfp import featurize
from src.featurization.maccs import add_maccs_features
from src.featurization.physchem import add_physchem_features
from src.featurization.mech_features import add_mechanism_flags
from src.featurization.docking import add_docking_scores
from src.featurization.structure import add_structure_features
from src.featurization.complex_features import add_complex_features
from src.models import XGBRegressor, EnsembleModel
from src.metrics.competition import nM_from_pIC50, comp_score
from src.eval.metrics import competition_score
from utils.config_loader import CFG
from utils.scaffold_split import scaffold_kfold
from utils.seed import set_seed

logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def oversample_extreme(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    low_thresh: float = 4.5,
    mid_thresh: float = 6.5,
    high_thresh: float = 7.5,
    target_sizes: Optional[dict] = None,
    sampler_type: str = "smoter",
):
    """Balance extreme pIC50 regions using SMOTER/SMOGN."""

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    low_mask = y < low_thresh
    mid_mask = (y >= low_thresh) & (y < mid_thresh)
    high_mask = (y >= mid_thresh) & (y < high_thresh)
    extra_high_mask = y >= high_thresh

    counts = {
        "low": int(low_mask.sum()),
        "mid": int(mid_mask.sum()),
        "high": int(high_mask.sum()),
        "extra_high": int(extra_high_mask.sum()),
    }
    if target_sizes is None:
        target_sizes = counts

    sampler_cls = SMOGN if sampler_type.lower() == "smogn" else SMOTER
    if sampler_cls is None:
        raise ImportError("imbalanced-learn is required for oversample_extreme")

    def _oversample_part(
        X_part: pd.DataFrame, y_part: pd.Series, target: int
    ) -> tuple[pd.DataFrame, pd.Series]:
        if len(X_part) == 0:
            return X_part, y_part
        if target <= len(X_part):
            return X_part.iloc[:target], y_part.iloc[:target]
        sampler = sampler_cls(random_state=42)
        X_res, y_res = X_part, y_part
        while len(X_res) < target:
            X_res, y_res = sampler.fit_resample(X_res, y_res)
        return X_res.iloc[:target], y_res.iloc[:target]

    X_low_res, y_low_res = _oversample_part(
        X[low_mask], y[low_mask], int(target_sizes["low"])
    )
    X_mid_res, y_mid_res = _oversample_part(
        X[mid_mask], y[mid_mask], int(target_sizes["mid"])
    )
    X_high_res, y_high_res = _oversample_part(
        X[high_mask], y[high_mask], int(target_sizes["high"])
    )
    X_extra_high_res, y_extra_high_res = _oversample_part(
        X[extra_high_mask], y[extra_high_mask], int(target_sizes["extra_high"])
    )

    X_res = pd.concat(
        [X_low_res, X_mid_res, X_high_res, X_extra_high_res], ignore_index=True
    )
    y_res = pd.concat(
        [y_low_res, y_mid_res, y_high_res, y_extra_high_res], ignore_index=True
    )
    return X_res.reset_index(drop=True), y_res.reset_index(drop=True)


def compute_kde_weights(y: pd.Series, bandwidth: float = 0.2) -> np.ndarray:
    """
    라벨 분포의 커널 밀도 추정을 이용해 역밀도 가중치를 계산한다.
    밀도가 높을수록 가중치는 작고, 밀도가 낮을수록 가중치는 크게 설정된다.
    bandwidth 값은 가우시안 커널의 대역폭이다.
    """
    kde = gaussian_kde(y.values, bw_method=bandwidth)
    densities = kde(y.values)
    densities = np.clip(densities, a_min=1e-8, a_max=None)
    weights = 1.0 / densities
    weights /= np.mean(weights)
    return weights


def compute_importance_weights(
    y_train: pd.Series,
    y_target: pd.Series,
    bandwidth: float = 0.2,
) -> np.ndarray:
    """
    훈련 분포와 목표 분포(예: raw 데이터 분포) 간의 밀도 비율을 통해 중요도 가중치를 계산한다.
    y_train: 현재 훈련 세트의 pIC50 값들
    y_target: 목표 분포(pIC50) 샘플. raw 데이터 전체 pIC50 시리즈를 전달한다.
    bandwidth: KDE 대역폭.
    """
    kde_train = gaussian_kde(y_train.values, bw_method=bandwidth)
    kde_target = gaussian_kde(y_target.values, bw_method=bandwidth)
    density_train = kde_train(y_train.values)
    density_target = kde_target(y_train.values)
    density_train = np.clip(density_train, a_min=1e-8, a_max=None)
    density_target = np.clip(density_target, a_min=1e-8, a_max=None)
    weights = density_target / density_train
    weights /= np.mean(weights)
    return weights


def load_raw_training_data(raw_dir: Path) -> pd.DataFrame:
    """Load CAS, ChEMBL and PubChem data and compute pIC50."""
    cas = pd.read_excel(
        raw_dir / "CAS_KPBMA_MAP3K5_IC50s.xlsx",
        sheet_name="MAP3K5 Ligand IC50s",
        header=1,
    )[["SMILES", "pX Value"]].rename(columns={"pX Value": "pIC50"})
    chembl = pd.read_csv(raw_dir / "ChEMBL_ASK1(IC50).csv", sep=";")[[
        "Smiles",
        "pChEMBL Value",
    ]].rename(columns={"Smiles": "SMILES", "pChEMBL Value": "pIC50"})
    pubchem = pd.read_csv(raw_dir / "Pubchem_ASK1.csv", low_memory=False)[[
        "SMILES",
        "Activity_Value",
    ]]
    pubchem["pIC50"] = -np.log10(pubchem["Activity_Value"] / 1_000_000)
    pubchem = pubchem[["SMILES", "pIC50"]]
    df = pd.concat([cas, chembl, pubchem], ignore_index=True)
    df = df.dropna(subset=["SMILES", "pIC50"])
    df["pIC50"] = pd.to_numeric(df["pIC50"], errors="coerce")
    df = df.dropna(subset=["pIC50"])
    return df


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise SMILES and remove duplicates."""
    df = df.copy()
    df["canonical_smiles"] = apply_standardization(df, smiles_col="SMILES")[
        "canonical_smiles"
    ]
    df = df.dropna(subset=["canonical_smiles"]).drop_duplicates(
        subset="canonical_smiles"
    )
    return df


def featurize_all(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full featurisation pipeline on ``df``."""
    df = featurize(df)
    df = add_maccs_features(df)
    df = add_physchem_features(df)
    df = add_mechanism_flags(df)
    df = add_docking_scores(df)
    df = add_structure_features(df)
    if CFG.get_hparam("use_complex_features"):
        df = add_complex_features(df)
    return df


def plot_distribution(df: pd.DataFrame, path: Path, column: str = "pIC50") -> None:
    """Plot a histogram of ``column`` and save to ``path``."""
    plt.figure()
    plt.hist(df[column], bins=40)
    plt.xlabel(column)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def tune_xgb_hyperparams(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_trials: int = 30,
    n_splits: int = 5,
) -> dict:
    """Tune ``XGBRegressor`` hyperparameters using Optuna with k-fold CV."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr_split, y_val_split = y.iloc[train_idx], y.iloc[val_idx]
            mask = y_tr_split.notna() & np.isfinite(y_tr_split)
            model = XGBRegressor(params)
            model.train(X_tr.loc[mask], y_tr_split.loc[mask], eval_set=(X_val, y_val_split))
            preds = model.predict(X_val)
            scores.append(r2_score(y_val_split, preds))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = project_root / "experiments"
    exp_dir.mkdir(exist_ok=True)
    with open(exp_dir / f"best_params_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    return best_params

def sanitise_labels(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["pIC50"].notna() & np.isfinite(df["pIC50"])
    dropped = len(df) - mask.sum()
    if dropped:
        logger.info("Dropped %d rows with imputed labels", dropped)
    return df.loc[mask].copy()


def train_and_evaluate(df: pd.DataFrame, args: argparse.Namespace) -> dict:
    df = sanitise_labels(df)
    smiles = df["canonical_smiles"].tolist()
    y = df["pIC50"].to_numpy()
    X = df.drop(columns=["pIC50"])

    pipe = build_feature_pipeline(X, {
        "imputer": args.imputer,
        "scaler": "standard",
        "exclude_cols": [c for c in X.columns if isinstance(c, int)]
    })
    X_proc = pipe.fit_transform(X)

    if args.split == "scaffold":
        splits = scaffold_kfold(smiles, n_splits=args.n_splits)
    else:
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        splits = kf.split(X_proc)

    metrics = []
    for fold, (tr, va) in enumerate(splits, 1):
        model = XGBRegressor(objective="reg:squarederror", random_state=42)
        model.fit(X_proc[tr], y[tr])
        pred = model.predict(X_proc[va])
        rmse = np.sqrt(mean_squared_error(y[va], pred))
        r2 = r2_score(y[va], pred)
        mae = mean_absolute_error(y[va], pred)
        score, A, B, rmse_nm = (None, None, None, None)
        if args.score_competition:
            score, A, B, rmse_nm = comp_score(
                nM_from_pIC50(y[va]), nM_from_pIC50(pred)
            )
        metrics.append({
            "fold": fold,
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "score": score,
            "A": A,
            "B": B,
            "rmse_nm": rmse_nm,
        })
    return {"folds": metrics}


def main(args: argparse.Namespace) -> None:
    """Train the regression model using the new raw-data workflow."""
    raw_dir = Path(CFG.get_path("raw_dir"))
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    raw_df = load_raw_training_data(raw_dir)
    raw_df.to_csv(raw_dir / "raw_input.csv", index=False)
    plot_distribution(raw_df, results_dir / "raw_input.png")

    processed_df = preprocess_input(raw_df)
    processed_df.to_csv(raw_dir / "processed_input.csv", index=False)
    plot_distribution(processed_df, results_dir / "processed_input.png")

    train_df, val_df = train_test_split(processed_df, test_size=args.val_size, random_state=42)
    train_df.to_csv(raw_dir / "processed_train.csv", index=False)
    val_df.to_csv(raw_dir / "processed_val.csv", index=False)
    plot_distribution(train_df, results_dir / "processed_train.png")
    plot_distribution(val_df, results_dir / "processed_val.png")

    train_feats = featurize_all(train_df)
    val_feats = featurize_all(val_df)
    feature_cols = [c for c in train_feats.columns if isinstance(c, int)] + ["tpsa","fsp3","docking_score","af_rmsd"]
    feature_cols = [c for c in feature_cols if c in train_feats.columns]
    scale_cols = [c for c in ["tpsa","fsp3","docking_score","af_rmsd"] if c in feature_cols]
    X_tr = train_feats[feature_cols].copy()
    y_tr = train_feats["pIC50"].copy()
    X_val = val_feats[feature_cols].copy()
    y_val = val_feats["pIC50"].copy()

    scaler=None
    if scale_cols:
        scaler=StandardScaler()
        X_tr.loc[:,scale_cols]=X_tr.loc[:,scale_cols].fillna(0.0)
        X_val.loc[:,scale_cols]=X_val.loc[:,scale_cols].fillna(0.0)
        X_tr.loc[:,scale_cols]=scaler.fit_transform(X_tr.loc[:,scale_cols])
        X_val.loc[:,scale_cols]=scaler.transform(X_val.loc[:,scale_cols])

    params=CFG.get_hparam("xgb_params") or {}
    if params.get("verbose") is True:
        params["verbose"]=False
    model=XGBRegressor(params)
    model.train(X_tr,y_tr)
    preds_val=model.predict(X_val)
    rmse=float((mean_squared_error(y_val,preds_val))**0.5)
    r2=float(r2_score(y_val,preds_val))
    mae=float(mean_absolute_error(y_val,preds_val))
    print(f"Validation RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")

    test_df=pd.read_csv(raw_dir/"test.csv")
    smiles_col="SMILES" if "SMILES" in test_df.columns else "Smiles"
    test_df=apply_standardization(test_df,smiles_col=smiles_col)
    test_feats=featurize_all(test_df)
    X_test=test_feats.loc[:,feature_cols].copy()
    if scale_cols and scaler is not None:
        X_test.loc[:,scale_cols]=X_test.loc[:,scale_cols].fillna(0.0)
        X_test.loc[:,scale_cols]=scaler.transform(X_test.loc[:,scale_cols])
    test_preds=model.predict(X_test)
    test_feats["pIC50_pred"]=test_preds
    plot_distribution(test_feats.rename(columns={"pIC50_pred":"pIC50"}),results_dir/"test_pred.png")

    base_dir=Path(CFG.get_path("submission_dir"))
    now=datetime.now(ZoneInfo("Asia/Seoul"))
    date_str=now.strftime("%Y%m%d")
    time_str=now.strftime("%H%M")
    sub_dir=base_dir/date_str/time_str
    sub_dir.mkdir(parents=True,exist_ok=True)
    file_ts=now.strftime("%Y%m%d_%H%M%S")
    submission_path=sub_dir/f"submission_{file_ts}.csv"
    out_df=pd.DataFrame({"ID":test_feats["ID"],"ASK1_IC50_nM":((10**(-test_preds))*1e9)})
    out_df.to_csv(submission_path,index=False)
    print(f"Test predictions saved to {submission_path}")

    return model
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASK1 IC50 regressor")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation set fraction")
    args = parser.parse_args()
    main(args)
