"""Extract simple interaction metrics from ASK1 complex structures.

This module parses PDB files containing ASK1 bound to partner proteins and
derives a handful of coarse structural descriptors.  The current
implementation computes three features:

``complex_interaction_residues``
    Number of ASK1 residues that have at least one heavy atom within 4Å of any
    atom in the partner chain.

``complex_hbond_count``
    Count of close N‒O/N‒N contacts (<3.5Å) between the two chains as a naive
    proxy for hydrogen bonds.

``complex_interface_area``
    Approximate solvent accessible surface area (SASA) lost upon complex
    formation for the ASK1 chain only.  It is estimated using the
    Shrake–Rupley algorithm provided by Biopython.  If SASA calculation fails,
    ``NaN`` is returned.

The features are identical for all ligands by default; however, a mapping
between compounds and specific PDB files can be supplied to extend the
functionality.  Missing structures or optional dependencies yield ``NaN``
values so downstream code can handle them gracefully.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from utils.config_loader import CFG

try:  # pragma: no cover - optional dependency
    from Bio.PDB import NeighborSearch, PDBParser, ShrakeRupley

    BIOPYTHON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    BIOPYTHON_AVAILABLE = False


def _nan_features() -> dict[str, float]:
    """Return a dictionary with NaN values for all complex features."""

    return {
        "complex_interaction_residues": np.nan,
        "complex_hbond_count": np.nan,
        "complex_interface_area": np.nan,
    }


def _compute_metrics(pdb_path: Path) -> dict[str, float]:
    """Compute interaction metrics for a given complex PDB file."""

    if not BIOPYTHON_AVAILABLE:
        return _nan_features()

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", pdb_path)
        model = next(structure.get_models())
        chains = list(model.get_chains())
        if len(chains) < 2:
            return _nan_features()
        chain_a, chain_b = chains[0], chains[1]

        atoms_a = [a for a in chain_a.get_atoms() if a.element != "H"]
        atoms_b = [a for a in chain_b.get_atoms() if a.element != "H"]
        ns = NeighborSearch(atoms_a + atoms_b)

        interacting: set[tuple[int, str]] = set()
        hbond_count = 0
        donors = {"N", "O"}
        for atom in atoms_a:
            close_atoms = ns.search(atom.coord, 4.0)
            for other in close_atoms:
                if other in atoms_a:
                    continue
                if other.get_parent().get_parent().id != chain_b.id:
                    continue
                interacting.add((atom.get_parent().id[1], chain_a.id))
                if (
                    atom.element in donors
                    and other.element in donors
                    and atom - other <= 3.5
                ):
                    hbond_count += 1

        # Estimate interface area for ASK1 (chain A)
        interface_area = np.nan
        try:
            sr = ShrakeRupley()
            chain_a_copy = chain_a.copy()
            sr.compute(chain_a_copy, level="C")
            asa_free = sum(res.sasa for res in chain_a_copy.get_residues())
            sr.compute(model)
            asa_complex = sum(res.sasa for res in chain_a.get_residues())
            interface_area = float(max(asa_free - asa_complex, 0.0))
        except Exception:
            interface_area = np.nan

        return {
            "complex_interaction_residues": float(len(interacting)),
            "complex_hbond_count": float(hbond_count),
            "complex_interface_area": interface_area,
        }
    except Exception:
        return _nan_features()


def add_complex_features(
    df: pd.DataFrame,
    *,
    pdb_dir: str | Path | None = None,
    mapping: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Add ASK1 complex interaction features to ``df``.

    Parameters
    ----------
    df:
        Input frame containing a ``canonical_smiles`` column.  The SMILES
        values are currently only used for optional mapping to a specific PDB
        file.
    pdb_dir:
        Directory containing complex PDB structures.  If ``None``, the path
        specified by ``complex_structures_dir`` in ``config/paths.yaml`` is
        used.
    mapping:
        Optional dictionary mapping canonical SMILES strings to PDB filenames.
        If omitted, a single default structure (``6EJL.pdb`` or the first
        available ``.pdb`` file) is used for all rows.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with three additional columns.  If structures are
        unavailable or Biopython is missing, the new columns contain ``NaN``.
    """

    feature_cols = list(_nan_features().keys())

    if not BIOPYTHON_AVAILABLE:
        for col in feature_cols:
            df[col] = np.nan
        return df

    structure_dir: Optional[Path]
    if pdb_dir is not None:
        structure_dir = Path(pdb_dir)
    else:
        cfg_path = CFG.get_path("complex_structures_dir")
        structure_dir = Path(cfg_path) if cfg_path else None

    if structure_dir is None or not structure_dir.exists():
        for col in feature_cols:
            df[col] = np.nan
        return df

    @lru_cache(maxsize=None)
    def compute_for(pdb_file: str) -> dict[str, float]:
        return _compute_metrics(structure_dir / pdb_file)

    def select_file(smiles: str) -> Path | None:
        if mapping and smiles in mapping:
            return structure_dir / mapping[smiles]
        fixed = structure_dir / "6EJL.pdb"
        if fixed.exists():
            return fixed
        files = list(structure_dir.glob("*.pdb"))
        return files[0] if files else None

    def calc(sm: str) -> pd.Series:
        path = select_file(sm)
        if path is None or not path.exists():
            return pd.Series(_nan_features())
        return pd.Series(compute_for(path.name))

    df = df.copy()
    res = df["canonical_smiles"].apply(calc)
    return pd.concat([df, res], axis=1)


__all__ = ["add_complex_features"]

