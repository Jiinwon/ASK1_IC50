from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas(desc="physchem")


def compute_physchem(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'tpsa': 0.0, 'fsp3': 0.0}
    return {
        'tpsa': Descriptors.TPSA(mol),
        'fsp3': Descriptors.FractionCSP3(mol)
    }


def add_physchem_features(df: pd.DataFrame) -> pd.DataFrame:
    phys = df['canonical_smiles'].progress_apply(compute_physchem)
    phys_df = pd.DataFrame(phys.tolist(), index=df.index)
    return pd.concat([df, phys_df], axis=1)
