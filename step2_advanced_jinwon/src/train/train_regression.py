"""Simplified training script implementing scaffold CV and competition metrics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.neighbors import NearestNeighbors

from src.data.impute import build_feature_pipeline
from src.data.splitters import compute_scaffolds
from src.data.standardize_smiles import apply_standardization
from src.data.impute_ic50 import (
    load_data,
    preprocess,
    impute_missing,
    save_outputs,
)
from src.data.feature_cache import load_or_create
from src.featurization.ecfp import featurize
from src.featurization.maccs import add_maccs_features
from src.featurization.physchem import add_physchem_features
from src.featurization.mech_features import add_mechanism_flags
from src.featurization.docking import add_docking_scores
from src.featurization.structure import add_structure_features
from src.models import XGBRegressor, EnsembleModel
from src.metrics.competition import nM_from_pIC50, comp_score
from src.eval.metrics import competition_score
from utils.config_loader import CFG
from utils.scaffold_split import scaffold_kfold
from utils.seed import set_seed

logger = logging.getLogger(__name__)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def oversample_extreme(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    lower: float = 4.5,
    upper: float = 6.5,
    target_size: int | str = "max",
):
    """Balance extreme pIC50 regions using SMOGN-style over/under-sampling."""

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    low_mask = y < lower
    high_mask = y > upper
    mid_mask = ~(low_mask | high_mask)

    n_low, n_mid, n_high = low_mask.sum(), mid_mask.sum(), high_mask.sum()
    if target_size == "max":
        target = max(n_low, n_mid, n_high)
    else:
        target = int(target_size)

    try:
        from imblearn.over_sampling import SMOTER as _SMOTEReg
    except Exception:  # pragma: no cover - fallback when SMOTER unavailable
        from imblearn.over_sampling import SMOTEN as _SMOTEReg  # type: ignore
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.utils import resample

    def _oversample_part(X_part: pd.DataFrame, y_part: pd.Series):
        if len(X_part) < 2:
            return X_part, y_part
        sampler = _SMOTEReg(random_state=42)
        X_res, y_res = X_part, y_part
        while len(X_res) < target:
            X_res, y_res = sampler.fit_resample(X_res, y_res)
        return X_res.iloc[:target], y_res.iloc[:target]

    X_low_res, y_low_res = _oversample_part(X[low_mask], y[low_mask])
    X_high_res, y_high_res = _oversample_part(X[high_mask], y[high_mask])

    X_mid_res, y_mid_res = X[mid_mask], y[mid_mask]
    if len(X_mid_res) > target:
        rus = RandomUnderSampler(sampling_strategy={0: target}, random_state=42)
        _, _ = rus.fit_resample(X_mid_res, np.zeros(len(X_mid_res)))
        sel_idx = rus.sample_indices_
        X_mid_res = X_mid_res.iloc[sel_idx]
        y_mid_res = y_mid_res.iloc[sel_idx]
    elif len(X_mid_res) < target and len(X_mid_res) > 0:
        X_mid_res, y_mid_res = resample(
            X_mid_res,
            y_mid_res,
            replace=True,
            n_samples=target,
            random_state=42,
        )

    X_res = pd.concat([X_low_res, X_mid_res, X_high_res], ignore_index=True)
    y_res = pd.concat([y_low_res, y_mid_res, y_high_res], ignore_index=True)
    return X_res.reset_index(drop=True), y_res.reset_index(drop=True)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data.csv"))
    parser.add_argument("--split", choices=["random", "scaffold"], default=CFG.get_hparam("split", "scaffold"))
    parser.add_argument("--n_splits", type=int, default=CFG.get_hparam("n_splits", 5))
    parser.add_argument("--imputer", choices=["iterative", "knn", "none"], default=CFG.get_hparam("imputer", "iterative"))
    parser.add_argument("--score_competition", action="store_true")
    args = parser.parse_args()

    best_params = CFG.get_hparam("xgb_params", {}) or {}
    if best_params.get("verbose") is True:
        best_params["verbose"] = False

    # Choose cross-validation splitter
    if args.split_method == "scaffold" and smiles is not None:
        groups = compute_scaffolds(smiles)
        splitter = GroupKFold(n_splits=args.cv_folds)
        split_iter = splitter.split(X, y, groups)
    elif args.split_method == "quantile":
        bins = pd.qcut(y, q=args.n_bins, labels=False, duplicates="drop")
        splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        split_iter = splitter.split(X, bins)
    else:
        splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        split_iter = splitter.split(X)

    r2_scores, mae_scores, rmse_scores = [], [], []
    score_list, A_list, B_list = [], [], []

    for fold, (train_idx, val_idx) in enumerate(split_iter, 1):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

        if args.oversample_extreme:
            X_tr, y_tr = oversample_extreme(X_tr, y_tr)
            X_tr, y_tr = oversample_quantile(X_tr, y_tr)
            X_tr, y_tr = smote_regression(X_tr, y_tr)
        elif args.oversample:
            X_tr, y_tr = oversample_quantile(X_tr, y_tr)
            X_tr, y_tr = smote_regression(X_tr, y_tr)

        # Standardize selected numeric features
        for df_split in (X_tr, X_val):
            df_split[scale_cols] = df_split[scale_cols].fillna(0.0)
        if scale_cols:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaler.fit(X_tr[scale_cols])
            X_tr[scale_cols] = scaler.transform(X_tr[scale_cols])
            X_val[scale_cols] = scaler.transform(X_val[scale_cols])

        X_tr, y_tr = clean_split(X_tr, y_tr, f"fold{fold} training")
        X_val, y_val = clean_split(X_val, y_val, f"fold{fold} validation")

        model = XGBRegressor(best_params)
        model.train(X_tr, y_tr)
        preds = model.predict(X_val)
        r2_scores.append(r2_score(y_val, preds))
        mae_scores.append(mean_absolute_error(y_val, preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        score, A, B = competition_score(y_val.values, preds)
        score_list.append(score)
        A_list.append(A)
        B_list.append(B)

    print(
        f"CV R2   : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}\n"
        f"CV MAE  : {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}\n"
        f"CV RMSE : {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}"
    )
    print(
        f"[SCORE] Score={np.mean(score_list):.6f} | A(normRMSE_IC50)={np.mean(A_list):.6f} | B(R2_IC50)={np.mean(B_list):.6f}"
    )

    # Train final model on the full dataset
    X_full, y_full = X.copy(), y.copy()
    if args.oversample_extreme:
        X_full, y_full = oversample_extreme(X_full, y_full)
        X_full, y_full = oversample_quantile(X_full, y_full)
        X_full, y_full = smote_regression(X_full, y_full)
    elif args.oversample:
        X_full, y_full = oversample_quantile(X_full, y_full)
        X_full, y_full = smote_regression(X_full, y_full)
    if scale_cols:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_full[scale_cols] = X_full[scale_cols].fillna(0.0)
        scaler.fit(X_full[scale_cols])
        X_full[scale_cols] = scaler.transform(X_full[scale_cols])
    else:
        scaler = None

    X_full, y_full = clean_split(X_full, y_full, "final training")

    ensemble_size = CFG.get_hparam("ensemble_size") or 1
    if ensemble_size > 1:
        model = EnsembleModel(best_params, n_models=ensemble_size)
    else:
        model = XGBRegressor(best_params)

    mask = y_full.notna() & ~np.isinf(y_full)
    X_full, y_full = X_full.loc[mask], y_full.loc[mask]
    model.train(X_full, y_full)

    # Prepare test set and generate predictions
    raw_dir = Path(CFG.get_path("raw_dir"))
    test_df = pd.read_csv(raw_dir / "test.csv")
    smiles_col = "SMILES" if "SMILES" in test_df.columns else "Smiles"
    test_df = apply_standardization(test_df, smiles_col=smiles_col)
    test_df = featurize(test_df)
    test_df = add_maccs_features(test_df)
    test_df = add_physchem_features(test_df)
    test_df = add_mechanism_flags(test_df)
    test_df = add_docking_scores(test_df)
    test_df = add_structure_features(test_df)

    X_test = test_df.loc[:, feature_cols].copy()
    if scale_cols and scaler is not None:
        X_test.loc[:, scale_cols] = X_test.loc[:, scale_cols].fillna(0.0)
        X_test.loc[:, scale_cols] = scaler.transform(X_test.loc[:, scale_cols])
    if ensemble_size > 1:
        test_preds, test_std = model.predict(X_test, return_std=True)
        test_df["prediction_std"] = test_std
    else:
        test_preds = model.predict(X_test)

    # ── 날짜/시각 하위 폴더(서울 기준) 생성 ─────────────────────────────
    base_dir = Path(CFG.get_path("submission_dir"))
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    date_str = now.strftime("%Y%m%d")     # 예: 20250807
    time_str = now.strftime("%H%M")       # 예: 1135
    sub_dir = base_dir / date_str / time_str
    sub_dir.mkdir(parents=True, exist_ok=True)
    file_ts = now.strftime("%Y%m%d_%H%M%S")
    submission_path = sub_dir / f"submission_{args.split_method}_{file_ts}.csv"
    out_df = pd.DataFrame(
        {"ID": test_df["ID"], "ASK1_IC50_nM": ((10 ** (-test_preds)) * 1e9)}
    )
    if ensemble_size > 1:
        out_df["pred_std"] = test_df["prediction_std"]
    out_df.to_csv(submission_path, index=False)
    print(f"Test predictions saved to {submission_path}")

    # ── 테스트 예측 분포 로그 ───────────────────────────────────────────
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    mean_pred, std_pred = float(np.mean(test_preds)), float(np.std(test_preds))
    plt.figure()
    plt.hist(test_preds, bins=40)
    plt.title(f"pIC50 predictions {mean_pred:.2f}±{std_pred:.2f}")
    plt.tight_layout()
    hist_path = results_dir / f"pIC50_pred_hist_{file_ts}.png"
    plt.savefig(hist_path)
    plt.close()
    print(f"Prediction histogram saved to {hist_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASK1 IC50 regressor")
    parser.add_argument(
        "--split-method",
        choices=["random", "quantile", "scaffold"],
        default="quantile",
        help="How to split data for cross-validation",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--n-bins", type=int, default=5, help="Quantile bins for stratification")
    parser.add_argument(
        "--min-ic50",
        type=float,
        default=None,
        help="Minimum IC50 value used when converting to pIC50",
    )
    parser.add_argument(
        "--use-log1p",
        action="store_true",
        help="Use log1p before log10 in IC50 to pIC50 conversion",
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Enable simple quantile and SMOTE-style oversampling",
    )
    parser.add_argument(
        "--oversample-extreme",
        action="store_true",
        help="Balance extreme pIC50 regions before standard oversampling",
    )
    args = parser.parse_args()
    main(args)