import numpy as np
from scipy.stats import pearsonr


def pic50_to_ic50_nM(p):
    p_arr = np.asarray(p, dtype=float)
    return np.power(10.0, -p_arr) * 1e9


def norm_rmse_ic50(y_true_pic50, y_pred_pic50):
    y_true_nM = pic50_to_ic50_nM(y_true_pic50)
    y_pred_nM = pic50_to_ic50_nM(y_pred_pic50)
    rmse = np.sqrt(np.mean((y_true_nM - y_pred_nM) ** 2))
    denom = np.max(y_true_nM) - np.min(y_true_nM)
    return rmse / denom if denom > 0 else float("nan")


def pearson_r2_ic50(y_true_pic50, y_pred_pic50):
    y_true_nM = pic50_to_ic50_nM(y_true_pic50)
    y_pred_nM = pic50_to_ic50_nM(y_pred_pic50)
    if np.var(y_true_nM) > 0 and np.var(y_pred_nM) > 0:
        r = pearsonr(y_true_nM, y_pred_nM)[0]
    else:
        r = 0.0
    return r ** 2


def competition_score(y_true_pic50, y_pred_pic50):
    A = norm_rmse_ic50(y_true_pic50, y_pred_pic50)
    B = pearson_r2_ic50(y_true_pic50, y_pred_pic50)
    return 0.4 * (1 - min(A, 1.0)) + 0.6 * B, A, B
