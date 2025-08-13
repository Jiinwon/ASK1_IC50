import numpy as np
import pandas as pd
from pathlib import Path
from utils.config_loader import CFG
from utils.chem_utils import standardize_smiles
from .impute_ic50 import load_data, preprocess, impute_missing, save_outputs


def IC50_to_pIC50(
    ic50_nM: pd.Series,
    *,
    min_value: float | None = None,
    percentile: float = 0.01,
    use_log1p: bool = False,
) -> pd.Series:
    """Convert IC50 (nM) values to pIC50 with a configurable floor.

    Parameters
    ----------
    ic50_nM:
        Series of IC50 values in nanomolar units.
    min_value:
        Absolute minimum value to clip IC50s at.  If ``None``, the
        ``percentile`` of the data is used instead.
    percentile:
        Quantile of the data used when ``min_value`` is ``None``.
    use_log1p:
        Apply ``np.log1p`` before the ``log10`` transform to lessen the
        impact of extreme low values.
    """

    ic50 = pd.to_numeric(ic50_nM, errors="coerce")
    if min_value is None:
        min_value = float(ic50.quantile(percentile))
    ic50 = ic50.clip(lower=min_value)
    if use_log1p:
        ic50 = np.log1p(ic50)
    return 9 - np.log10(ic50)


def load_and_preprocess(
    *,
    min_ic50: float | None = None,
    percentile: float = 0.01,
    use_log1p: bool = False,
) -> pd.DataFrame:
    """Load datasets, impute IC50, and return a unified table."""

    cas, chembl, pubchem = load_data()
    cas, chembl, pubchem = preprocess(cas, chembl, pubchem)
    df, counts = impute_missing(cas, chembl, pubchem)
    save_outputs(df, counts)

    df = df.dropna(subset=["SMILES", "IC50_nM"])
    df = df.drop_duplicates(subset="SMILES")
    df["canonical_smiles"] = df["SMILES"].apply(standardize_smiles)
    df["pIC50"] = IC50_to_pIC50(
        df["IC50_nM"], min_value=min_ic50, percentile=percentile, use_log1p=use_log1p
    )
    return df[["SMILES", "canonical_smiles", "IC50_nM", "pIC50"]]
