from __future__ import annotations

from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

from utils.config_loader import CFG
from .baseline_loader import load_and_preprocess
from src.featurization.ecfp import featurize
from src.featurization.maccs import add_maccs_features
from src.featurization.physchem import add_physchem_features
from src.featurization.mech_features import add_mechanism_flags
from src.featurization.docking import add_docking_scores
from src.featurization.structure import add_structure_features
from src.featurization.complex_features import add_complex_features


def _build_features(*, min_ic50: float | None = None, use_log1p: bool = False) -> pd.DataFrame:
    """Run the full featurization pipeline."""
    steps = [
        ("load", lambda: load_and_preprocess(min_ic50=min_ic50, use_log1p=use_log1p)),
        ("featurize", featurize),
        ("maccs", add_maccs_features),
        ("physchem", add_physchem_features),
        ("mechanism", add_mechanism_flags),
        ("docking", add_docking_scores),
        ("structure", add_structure_features),
    ]
    if CFG.get_hparam("use_complex_features"):
        steps.append(("complex", add_complex_features))
    df: pd.DataFrame | None = None
    pbar = tqdm(total=len(steps))
    for i, (desc, fn) in enumerate(steps):
        if i == 0:
            df = fn()
        else:
            assert df is not None
            df = fn(df)
        pbar.set_description(desc)
        pbar.update(1)
    pbar.close()
    assert df is not None
    return df


def load_or_create(*, min_ic50: float | None = None, use_log1p: bool = False) -> pd.DataFrame:
    """Load cached features or create them if missing."""
    feats_dir = Path(CFG.get_path("featurized_dir"))
    feats_dir.mkdir(parents=True, exist_ok=True)

    name = CFG.get_hparam("feature_sheet") or "feature_1"
    path = feats_dir / f"{name}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        # Convert any integer-like column names back to integers for
        # downstream code that expects them.
        new_cols = []
        for c in df.columns:
            if isinstance(c, str) and c.isdigit():
                new_cols.append(int(c))
            else:
                new_cols.append(c)
        df.columns = new_cols
        return df

    df = _build_features(min_ic50=min_ic50, use_log1p=use_log1p)

    # Parquet requires string column names.  Some of our features use
    # integer names (e.g. ECFP/MACCS bits), so convert them to strings
    # for storage and then back to integers for the returned frame.
    df_storage = df.copy()
    df_storage.columns = [str(c) for c in df_storage.columns]
    df_storage.to_parquet(path, index=False)

    return df
