import numpy as np
import pandas as pd
from pathlib import Path

from utils.config_loader import CFG

try:
    from colabfold.batch import run as cf_run  # type: ignore
    from Bio.PDB import PDBParser, Superimposer
    from rdkit import Chem
    from rdkit.Chem import AllChem
    ALPHAFOLD_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    ALPHAFOLD_AVAILABLE = False


def _predict_complex(smiles: str, fasta_path: str, tmpdir: Path) -> Path | None:
    """Run a minimal ColabFold prediction for the ASK1-ligand complex."""
    if not ALPHAFOLD_AVAILABLE:
        return None
    try:
        seq = Path(fasta_path).read_text().splitlines()[1]
        fasta_file = tmpdir / "ask1.fasta"
        fasta_file.write_text(f">ASK1\n{seq}\n")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        lig_pdb = tmpdir / "lig.pdb"
        Chem.MolToPDBFile(mol, str(lig_pdb))
        # The ColabFold batch API will output `ranked_0.pdb` in tmpdir
        cf_run(str(fasta_file), str(tmpdir), "alphafold2_multimer" )
        complex_pdb = tmpdir / "ranked_0.pdb"
        if complex_pdb.exists():
            return complex_pdb
    except Exception:
        return None
    return None


def _compute_rmsd(ref_path: Path, mobile_path: Path) -> float:
    parser = PDBParser(QUIET=True)
    ref = parser.get_structure("ref", ref_path)[0]
    mobile = parser.get_structure("mobile", mobile_path)[0]
    ref_atoms = [a for a in ref.get_atoms() if a.element != "H"]
    mob_atoms = [a for a in mobile.get_atoms() if a.element != "H"]
    if len(ref_atoms) != len(mob_atoms):
        return np.nan
    sup = Superimposer()
    sup.set_atoms(ref_atoms, mob_atoms)
    return float(sup.rms)


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Predict structural change upon ligand binding using AlphaFold."""
    fasta = CFG.get_path("ask1_fasta")
    unbound_pdb = CFG.get_path("receptor_pdbqt").replace(".pdbqt", ".pdb")
    if fasta is None or not Path(fasta).exists():
        df["af_rmsd"] = np.nan
        return df
    unbound_pdb = Path(unbound_pdb)
    if not unbound_pdb.exists():
        df["af_rmsd"] = np.nan
        return df

    def calc(sm: str):
        if not ALPHAFOLD_AVAILABLE:
            return np.nan
        try:
            with tempfile.TemporaryDirectory() as td:
                tmpdir = Path(td)
                complex_pdb = _predict_complex(sm, fasta, tmpdir)
                if complex_pdb is None:
                    return np.nan
                return _compute_rmsd(unbound_pdb, complex_pdb)
        except Exception:
            return np.nan

    import tempfile
    from tqdm.auto import tqdm

    tqdm.pandas(desc="alphafold")
    df["af_rmsd"] = df["canonical_smiles"].progress_apply(calc)
    return df