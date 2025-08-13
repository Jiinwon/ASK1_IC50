"""Competition specific scoring utilities."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def pIC50_from_nM(nm):
    nm = np.asarray(nm, dtype=float)
    nm = np.clip(nm, 1e-10, None)
    return 9.0 - np.log10(nm)


def nM_from_pIC50(pic50):
    return np.power(10.0, 9.0 - np.asarray(pic50, dtype=float))


def comp_score(y_true_nm, y_pred_nm):
    y_true_nm = np.asarray(y_true_nm, dtype=float)
    y_pred_nm = np.asarray(y_pred_nm, dtype=float)
    # A: Normalized RMSE in IC50(nM)
    rmse = np.sqrt(np.mean((y_true_nm - y_pred_nm) ** 2))
    denom = max(np.ptp(y_true_nm), 1e-12)
    A = rmse / denom
    A = min(A, 1.0)
    # B: Pearson^2 between pIC50 true/pred
    r, _ = pearsonr(pIC50_from_nM(y_true_nm), pIC50_from_nM(y_pred_nm))
    B = float(r**2)
    return 0.4 * (1.0 - A) + 0.6 * B, A, B, rmse

def competition_metric(preds, dtrain):
    """Custom XGBoost evaluation metric that logs competition scores.

    Parameters
    ----------
    preds : np.ndarray
        Predicted values in pIC50 space.
    dtrain : xgb.DMatrix
        Training matrix holding the true pIC50 labels.

    Returns
    -------
    tuple[str, float]
        Name of the metric and the competition score.
    """
    y_true = dtrain.get_label()
    score, A, B, _ = comp_score(nM_from_pIC50(y_true), nM_from_pIC50(preds))
    # Mirror the typical XGBoost log format while exposing additional details
    print(f"A:{A:.5f} B:{B:.5f} Score:{score:.5f}")
    return "Score", score