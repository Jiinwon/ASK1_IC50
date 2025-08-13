from rdkit import Chem
from rdkit.Chem import inchi
import pandas as pd

df = pd.read_csv("/home1/won0316/DACON/JUMP_AI_2025_EST/data/test.csv")  # columns: ID, Smiles
def to_inchikey(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return inchi.MolToInchiKey(m)
df["InChIKey"] = df["Smiles"].apply(to_inchikey)
df[["ID","InChIKey"]].to_csv("test_warn/batch_inchikeys.tsv", sep="\t", index=False)
