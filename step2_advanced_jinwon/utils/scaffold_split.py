"""Scaffold-based cross-validation utilities."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Generator, Iterable, List, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


def _murcko_scaffold(smiles: str) -> str:
    """Return the Bemis-Murcko scaffold for ``smiles`` as canonical SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold) if scaffold is not None else ""


def scaffold_kfold(
    smiles: Sequence[str],
    n_splits: int = 5,
    seed: int = 42,
    min_scaffold_size: int = 5,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Yield train/validation indices for scaffold-based CV.

    Parameters
    ----------
    smiles:
        Sequence of SMILES strings.
    n_splits:
        Number of folds.
    seed:
        Random seed controlling scaffold shuffling.
    min_scaffold_size:
        Minimum number of molecules required for a scaffold to be considered on
        its own; smaller scaffolds are grouped together.
    """

    rng = np.random.default_rng(seed)

    scaffolds: defaultdict[str, List[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles):
        scaff = _murcko_scaffold(smi)
        scaffolds[scaff].append(idx)

    # Group scaffolds by size and shuffle to distribute evenly
    scaffold_items = list(scaffolds.items())
    rng.shuffle(scaffold_items)
    scaffold_items.sort(key=lambda kv: len(kv[1]), reverse=True)

    folds: List[List[int]] = [[] for _ in range(n_splits)]

    # Pre-compute fingerprints for similarity checks
    scaffold_fps = {}
    for scaff, idxs in scaffold_items:
        mol = Chem.MolFromSmiles(scaff)
        if mol:
            scaffold_fps[scaff] = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        else:
            scaffold_fps[scaff] = None

    fold_fps: List[List[object]] = [[] for _ in range(n_splits)]

    for scaff, idxs in scaffold_items:
        # find fold with smallest size that keeps tanimoto < 0.4 wrt existing scaffolds
        best_fold = None
        for fold_idx in np.argsort([len(f) for f in folds]):
            fps = fold_fps[fold_idx]
            fp = scaffold_fps.get(scaff)
            if fp is not None and fps:
                sims = [DataStructs.TanimotoSimilarity(fp, other) for other in fps]
                if max(sims) >= 0.4:
                    continue
            best_fold = fold_idx
            break
        if best_fold is None:
            best_fold = int(np.argmin([len(f) for f in folds]))
        folds[best_fold].extend(idxs)
        if scaffold_fps.get(scaff) is not None:
            fold_fps[best_fold].append(scaffold_fps[scaff])

    all_indices = set(range(len(smiles)))
    for k in range(n_splits):
        val_idx = np.array(folds[k], dtype=int)
        train_idx = np.array(sorted(all_indices - set(val_idx)), dtype=int)
        yield train_idx, val_idx
