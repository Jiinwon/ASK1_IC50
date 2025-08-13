import pandas as pd
from pathlib import Path
from utils.config_loader import CFG


def load_all():
    raw = Path(CFG.get_path('raw_dir'))
    cas = pd.read_excel(raw / 'CAS_KPBMA_MAP3K5_IC50s.xlsx')
    chembl = pd.read_csv(raw / 'ChEMBL_ASK1(IC50).csv', sep=';')
    pubchem = pd.read_csv(raw / 'Pubchem_ASK1.csv')
    return cas, chembl, pubchem


def load_integrated() -> pd.DataFrame:
    """Load the pre-merged dataset if available."""
    raw = Path(CFG.get_path('raw_dir'))
    return pd.read_excel(raw / 'Integrated_data.xlsx')

if __name__ == '__main__':
    cas, chembl, pubchem = load_all()
    print('Loaded:', len(cas), len(chembl), len(pubchem))
