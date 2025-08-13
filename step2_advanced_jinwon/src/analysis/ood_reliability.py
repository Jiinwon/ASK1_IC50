"""Tools for analysing out-of-distribution behaviour and prediction reliability."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _ecfp_array(smiles: Iterable[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=int))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=True)
        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return np.asarray(fps)


def knn_ood_distance(
    train_features: np.ndarray | Iterable[str],
    test_features: np.ndarray | Iterable[str],
    k: int = 5,
) -> np.ndarray:
    """Compute kNN distance using the same features used for model training.

    Parameters
    ----------
    train_features, test_features : array-like or Iterable[str]
        Either pre-computed feature arrays (e.g. fingerprints) or SMILES
        strings from which ECFP fingerprints will be generated.
    k : int, optional
        Number of neighbours to average over.
    """

    if isinstance(train_features, np.ndarray):
        train_fp = np.asarray(train_features)
    else:
        train_fp = _ecfp_array(train_features)

    if isinstance(test_features, np.ndarray):
        test_fp = np.asarray(test_features)
    else:
        test_fp = _ecfp_array(test_features)

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(train_fp)
    dists, _ = nn.kneighbors(test_fp, return_distance=True)
    return dists.mean(axis=1)


def reliability_plots(y_true: np.ndarray, y_pred: np.ndarray, ood_dist: np.ndarray, out_dir: Path) -> None:
    """Generate simple OOD and reliability plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    abs_err = np.abs(y_true - y_pred)

    plt.figure()
    plt.hist(ood_dist, bins=50)
    plt.xlabel("kNN Tanimoto distance")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "ood_knn_hist.png")
    plt.close()

    plt.figure()
    plt.scatter(ood_dist, abs_err, alpha=0.6)
    plt.xlabel("OOD distance")
    plt.ylabel("|error|")
    plt.tight_layout()
    plt.savefig(out_dir / "ood_vs_abs_error.png")
    plt.close()

    # Reliability diagram: residual variance vs OOD distance
    bins = np.quantile(ood_dist, np.linspace(0, 1, 6))
    indices = np.digitize(ood_dist, bins[1:-1])
    means = [abs_err[indices == i].mean() for i in range(len(bins))]
    plt.figure()
    plt.plot(range(len(means)), means, marker="o")
    plt.xlabel("OOD bin")
    plt.ylabel("mean |error|")
    plt.tight_layout()
    plt.savefig(out_dir / "reliability_diagram.png")
    plt.close()

    pd.DataFrame({"ood_distance": ood_dist, "abs_error": abs_err}).to_csv(
        out_dir / "ood_summary.csv", index=False
    )
