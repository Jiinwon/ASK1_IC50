import pandas as pd
from tqdm.auto import tqdm
from utils.config_loader import CFG
from utils.chem_utils import compute_maccs

tqdm.pandas(desc="maccs")


def add_maccs_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MACCS keys and append them as numerical columns."""
    start = CFG.get_hparam('ecfp_bits')
    fps = df['canonical_smiles'].progress_apply(compute_maccs)
    cols = list(range(start, start + 167))
    fp_df = pd.DataFrame(fps.tolist(), index=df.index, columns=cols)
    return pd.concat([df, fp_df], axis=1)
