"""Utilities for dataset splitting with stratification and scaffold grouping."""

from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
)


def quantile_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    n_bins: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data using quantile-based stratification on ``y``."""
    bins = pd.qcut(y, q=n_bins, duplicates="drop")
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=bins,
    )


def _scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def compute_scaffolds(smiles: Iterable[str]) -> pd.Series:
    """Return Bemis-Murcko scaffold identifiers for ``smiles``."""
    return pd.Series([_scaffold(s) for s in smiles], index=getattr(smiles, "index", None))


def scaffold_group_split(
    df: pd.DataFrame,
    *,
    smiles_col: str = "canonical_smiles",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return train/validation indices based on Bemis-Murcko scaffolds."""
    scaffolds = df[smiles_col].map(_scaffold)
    gss = GroupShuffleSplit(
        test_size=test_size, n_splits=1, random_state=random_state
    )
    idx = np.arange(len(df))
    train_idx, val_idx = next(gss.split(idx, groups=scaffolds))
    return train_idx, val_idx


def make_cv_splitter(
    method: str,
    y: pd.Series | None = None,
    groups: Iterable[str] | None = None,
    *,
    n_splits: int = 5,
    n_bins: int = 5,
    random_state: int = 42,
):
    """Create an appropriate CV splitter based on ``method``."""
    if method == "scaffold":
        return GroupKFold(n_splits=n_splits)
    if method == "stratified":
        assert y is not None
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    # default random split
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
