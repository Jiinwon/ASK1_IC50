from utils.chem_utils import standardize_smiles
from tqdm.auto import tqdm

tqdm.pandas(desc="standardize")

def apply_standardization(df, smiles_col='Smiles'):
    """Apply SMILES standardization with a progress bar."""
    df['canonical_smiles'] = df[smiles_col].progress_apply(standardize_smiles)
    return df
