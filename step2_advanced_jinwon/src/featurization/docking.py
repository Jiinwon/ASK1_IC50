import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utils.config_loader import CFG

import shutil

try:
    from vina import Vina

    VINA_AVAILABLE = True
except Exception:
    VINA_AVAILABLE = False

GNINA_BIN = shutil.which("gnina")


def compute_vina_score(smiles: str, receptor: str) -> float:
    """Compute docking score with AutoDock Vina if available."""
    if not VINA_AVAILABLE:
        return np.nan
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import tempfile
        import os

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = os.path.join(tmpdir, "lig.pdb")
            Chem.MolToPDBFile(mol, pdb_path)
            pdbqt_path = os.path.join(tmpdir, "lig.pdbqt")
            try:
                import openbabel.pybel as pybel

                lig = next(pybel.readfile("pdb", pdb_path))
                lig.write("pdbqt", pdbqt_path, overwrite=True)
            except Exception:
                return np.nan
            v = Vina(sf_name="vina")
            v.set_receptor(receptor)
            v.set_ligand_from_file(pdbqt_path)
            v.compute_vina_maps(center=[0, 0, 0], box_size=[20, 20, 20])
            result = v.score()
            return result.get("vina_affinity", np.nan)
    except Exception:
        return np.nan


def compute_gnina_score(smiles: str, receptor: str) -> float:
    """Compute docking score with GNINA if installed."""
    if GNINA_BIN is None:
        return np.nan
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import tempfile
        import os

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = os.path.join(tmpdir, "lig.pdb")
            Chem.MolToPDBFile(mol, pdb_path)
            out_path = os.path.join(tmpdir, "out.sdf")
            cmd = [GNINA_BIN, "-r", receptor, "-l", pdb_path, "-o", out_path]
            os.system(" ".join(cmd))
            if not os.path.exists(out_path):
                return np.nan
            with open(out_path) as f:
                for line in f:
                    if line.startswith("> <minimizedAffinity>"):
                        next(f)
                        return float(next(f).strip())
        return np.nan
    except Exception:
        return np.nan


def add_docking_scores(df: pd.DataFrame, method: str = "gnina") -> pd.DataFrame:
    """Add docking affinity scores using the specified engine."""
    receptor = CFG.get_path("receptor_pdbqt")
    if receptor is None:
        df["docking_score"] = np.nan
        return df
    if method == "gnina" and GNINA_BIN is not None:
        scorer = lambda sm: compute_gnina_score(sm, receptor)
    else:
        scorer = lambda sm: compute_vina_score(sm, receptor)
    tqdm.pandas(desc="docking")
    df["docking_score"] = df["canonical_smiles"].progress_apply(scorer)
    return df
