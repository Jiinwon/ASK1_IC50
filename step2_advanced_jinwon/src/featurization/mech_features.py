from rdkit import Chem
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas(desc="mech")

# Simple SMARTS patterns as heuristics for mechanistic classes
THIOREDOXIN_SMARTS = '[C;D2]=[C;D2]-[C;D2]=O'
TRIAZOLE_SMARTS = 'n1nnc([nH])n1'
PHOSPHO_SMARTS = 'P(=O)(O)O'
NITRILE_SMARTS = 'C#N'

trx_pat = Chem.MolFromSmarts(THIOREDOXIN_SMARTS)
triazole_pat = Chem.MolFromSmarts(TRIAZOLE_SMARTS)
phospho_pat = Chem.MolFromSmarts(PHOSPHO_SMARTS)
nitrile_pat = Chem.MolFromSmarts(NITRILE_SMARTS)


def add_mechanism_flags(df: pd.DataFrame) -> pd.DataFrame:
    def has_trx(sm):
        m = Chem.MolFromSmiles(sm)
        return int(m.HasSubstructMatch(trx_pat)) if m else 0

    def has_hinge(sm):
        m = Chem.MolFromSmiles(sm)
        return int(m.HasSubstructMatch(triazole_pat)) if m else 0

    def has_phospho(sm):
        m = Chem.MolFromSmiles(sm)
        return int(m.HasSubstructMatch(phospho_pat)) if m else 0

    def has_nitrile(sm):
        m = Chem.MolFromSmiles(sm)
        return int(m.HasSubstructMatch(nitrile_pat)) if m else 0

    df['thioredoxin_binding'] = df['canonical_smiles'].progress_apply(has_trx)
    df['atp_competitive'] = df['canonical_smiles'].progress_apply(has_hinge)
    df['binds_14_3_3'] = df['canonical_smiles'].progress_apply(has_phospho)
    df['fbxo21_ubiquitin'] = df['canonical_smiles'].progress_apply(has_nitrile)
    return df
