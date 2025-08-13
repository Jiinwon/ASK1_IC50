import pandas as pd
from pathlib import Path
from utils.config_loader import CFG


def read_raw(filename: str) -> pd.DataFrame:
    path = Path(CFG.get_path('raw_dir')) / filename
    if filename.endswith('.xlsx'):
        return pd.read_excel(path)
    return pd.read_csv(path)


def save_interim(df: pd.DataFrame, name: str):
    path = Path(CFG.get_path('interim_dir')) / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
