from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen


def _to_mol(smiles: str):
    if pd.isna(smiles) or not isinstance(smiles, str):
        return None
    s = smiles.strip()
    if not s:
        return None
    return Chem.MolFromSmiles(s)


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing descriptor values using RDKit and nearest neighbors."""
    mols = df.get("canonical_smiles", df.get("Smiles"))
    df = df.copy()
    df["_mol"] = mols.apply(_to_mol)

    # Molecular Weight
    mask = df.get("Molecular Weight").isna()
    df.loc[mask, "Molecular Weight"] = df.loc[mask, "_mol"].apply(
        lambda m: Descriptors.MolWt(m) if m else np.nan
    )

    # AlogP
    mask = df.get("AlogP").isna()
    df.loc[mask, "AlogP"] = df.loc[mask, "_mol"].apply(
        lambda m: Crippen.MolLogP(m) if m else np.nan
    )

    # Use Standard Value only when Standard Type is IC50
    if "Standard Type" in df.columns and "Standard Value" in df.columns:
        df.loc[df["Standard Type"] != "IC50", "Standard Value"] = np.nan

        # Prepare fingerprints
        fps = [
            AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None
            for m in df["_mol"]
        ]
        df["_fp"] = fps

        known_idx = df.index[df["Standard Value"].notna()]
        known_fps = [fps[i] for i in known_idx]

        for idx, row in df[df["Standard Value"].isna() & df["_fp"].notna()].iterrows():
            fp = row["_fp"]
            sims = [
                DataStructs.TanimotoSimilarity(fp, kfp) if kfp is not None else 0
                for kfp in known_fps
            ]
            if sims:
                best = known_idx[int(np.argmax(sims))]
                df.at[idx, "Standard Value"] = df.at[best, "Standard Value"]

    # Ligand efficiency metrics
    ic50 = df.get("Standard Value")
    if ic50 is not None:
        minus_log_ic50 = -np.log10(ic50.astype(float) / 1_000_000_000)
        heavy_atoms = df["_mol"].apply(lambda m: m.GetNumHeavyAtoms() if m else np.nan)
        psa = df.get("Polar Surface Area", df.get("tpsa"))

        if "Ligand Efficiency LE" in df.columns:
            df["Ligand Efficiency LE"] = df["Ligand Efficiency LE"].fillna(
                minus_log_ic50 / heavy_atoms
            )
        if "Ligand Efficiency LLE" in df.columns:
            df["Ligand Efficiency LLE"] = df["Ligand Efficiency LLE"].fillna(
                minus_log_ic50 - df["AlogP"]
            )
        if "Ligand Efficiency SEI" in df.columns and psa is not None:
            df["Ligand Efficiency SEI"] = df["Ligand Efficiency SEI"].fillna(
                minus_log_ic50 / (psa / 100)
            )

    return df.drop(columns=["_mol", "_fp"], errors="ignore")
