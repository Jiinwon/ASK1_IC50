#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training script for an MLPRegressor ensemble."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from utils.config_loader import CFG
from utils.seed import set_seed
from src.data.feature_cache import load_or_create
from src.data.standardize_smiles import apply_standardization
from src.featurization.ecfp import featurize
from src.featurization.maccs import add_maccs_features
from src.featurization.physchem import add_physchem_features
from src.featurization.mech_features import add_mechanism_flags
from src.featurization.docking import add_docking_scores
from src.featurization.structure import add_structure_features
from src.featurization.complex_features import add_complex_features
from src.models import MLPEnsembleModel


def main():
    set_seed(42)
    df = load_or_create()
    feature_cols = [c for c in df.columns if isinstance(c, int)] + [
        "tpsa",
        "fsp3",
        "docking_score",
        "af_rmsd",
        "complex_interaction_residues",
        "complex_hbond_count",
        "complex_interface_area",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    scale_cols = [
        c
        for c in [
            "tpsa",
            "fsp3",
            "docking_score",
            "af_rmsd",
            "complex_interaction_residues",
            "complex_hbond_count",
            "complex_interface_area",
        ]
        if c in feature_cols
    ]

    X = df[feature_cols]
    y = df["pIC50"]


    def oversample_quantile(X_df: pd.DataFrame, y_sr: pd.Series, *, bins: int = 10):
        """Oversample to obtain roughly uniform target distribution."""
        q = pd.qcut(y_sr, q=bins, duplicates="drop")
        max_size = q.value_counts().max()
        parts = []
        for level in q.unique():
            idx = q == level
            part = pd.concat(
                [X_df[idx], y_sr[idx]], axis=1
            ).sample(max_size, replace=True, random_state=42)
            parts.append(part)
        resampled = pd.concat(parts)
        return resampled[X_df.columns], resampled[y_sr.name]

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, y_tr = oversample_quantile(X_tr, y_tr)

    for df_split in (X_tr, X_val):
        df_split[scale_cols] = df_split[scale_cols].fillna(0.0)
    if scale_cols:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_tr[scale_cols])
        X_tr[scale_cols] = scaler.transform(X_tr[scale_cols])
        X_val[scale_cols] = scaler.transform(X_val[scale_cols])

    model = MLPEnsembleModel({}, n_models=3)
    model.train(X_tr, y_tr)

    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    print(f"Validation R2: {r2:.4f}")

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
    if CFG.get_hparam("use_complex_features"):
        test_df = add_complex_features(test_df)

    X_test = test_df[feature_cols]
    X_test[scale_cols] = X_test[scale_cols].fillna(0.0)
    if scale_cols:
        X_test[scale_cols] = scaler.transform(X_test[scale_cols])
    test_preds = model.predict(X_test)

    now = datetime.now()
    sub_dir = Path("submission") / now.strftime("%Y-%m-%d")
    sub_dir.mkdir(parents=True, exist_ok=True)
    submission_path = sub_dir / f"submission_{now.strftime('%H%M%S')}.csv"

    out_df = pd.DataFrame({"ID": test_df["ID"], "ASK1_IC50_nM": ((10 **(-test_preds)) * 1e9)})
    out_df.to_csv(submission_path, index=False)
    print(f"Test predictions saved to {submission_path}")

    return model, r2


if __name__ == "__main__":
    main()
