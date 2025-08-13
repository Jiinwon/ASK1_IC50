from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde  # 추가
from scipy.stats import gaussian_kde  # 추가

def IC50_to_pIC50(ic50_nM: pd.Series) -> pd.Series:
    ic50_nM = pd.to_numeric(ic50_nM, errors="coerce")
    ic50_nM = ic50_nM.clip(lower=1e-10)
    return 9 - np.log10(ic50_nM)

def _clean_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan).dropna()
def _clean_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan).dropna()

def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir.parent / "data"
    result_dir = Path(__file__).resolve().parent / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    # 1) Raw training
    # 1) Raw training
    raw_path = data_dir / "Pubchem_ASK1.csv"
    raw_df = pd.read_csv(raw_path, low_memory=False)
    raw_df = raw_df[["Activity_Value"]].rename(columns={"Activity_Value": "IC50_nM"})
    raw_df = raw_df.dropna(subset=["IC50_nM"])
    raw_df["pIC50"] = IC50_to_pIC50(raw_df["IC50_nM"])
    s_raw = _clean_series(raw_df["pIC50"])
    s_raw = _clean_series(raw_df["pIC50"])

    # 2) Preprocessed full
    # 2) Preprocessed full
    processed_path = data_dir / "Pubchem_ASK1_imputed.csv"
    processed_df = pd.read_csv(processed_path)
    processed_df["pIC50"] = IC50_to_pIC50(processed_df["IC50_nM"])
    s_full = _clean_series(processed_df["pIC50"])
    s_full = _clean_series(processed_df["pIC50"])

    # 3) Train/Val split (더미 X 그대로)
    # 3) Train/Val split (더미 X 그대로)
    y = processed_df["pIC50"]
    X_dummy = processed_df.drop(columns=["IC50_nM", "pIC50"])
    X_tr, X_val, y_tr, y_val = train_test_split(X_dummy, y, test_size=0.2, random_state=42)
    s_tr = _clean_series(y_tr)
    s_val = _clean_series(y_val)
    X_dummy = processed_df.drop(columns=["IC50_nM", "pIC50"])
    X_tr, X_val, y_tr, y_val = train_test_split(X_dummy, y, test_size=0.2, random_state=42)
    s_tr = _clean_series(y_tr)
    s_val = _clean_series(y_val)

    # 4) Predictions on test.csv
    pred_path = base_dir / "submissions" / "20250807" / "1652" / "submission.csv"
    pred_df = pd.read_csv(pred_path)
    pred_df["pIC50"] = IC50_to_pIC50(pred_df["ASK1_IC50_nM"])
    s_pred = _clean_series(pred_df["pIC50"])

    datasets = [
        ("Raw Training", s_raw),
        ("Processed Full", s_full),
        ("Processed Train", s_tr),
        ("Processed Validation", s_val),
        ("Predicted Test", s_pred),
    ]

    # 공통 x축 (5% 마진)
    all_vals = pd.concat([s for _, s in datasets], ignore_index=True)
    x_min, x_max = float(all_vals.min()), float(all_vals.max())
    if np.isfinite(x_min) and np.isfinite(x_max) and x_min < x_max:
        pad = 0.05 * (x_max - x_min)
        x_min, x_max = x_min - pad, x_max + pad
    else:
        x_min, x_max = 0.0, 10.0
    x_grid = np.linspace(x_min, x_max, 512)

    # 2행×5열: 위 히스토그램(빈도), 아래 KDE(밀도)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), tight_layout=True)
    bins = 50

    # 위 행: 히스토그램
    for i, (name, s) in enumerate(datasets):
        ax = axes[0, i]
        ax.hist(s, bins=bins, edgecolor="black")
        ax.set_title(f"{name}\nHistogram")
    s_pred = _clean_series(pred_df["pIC50"])

    datasets = [
        ("Raw Training", s_raw),
        ("Processed Full", s_full),
        ("Processed Train", s_tr),
        ("Processed Validation", s_val),
        ("Predicted Test", s_pred),
    ]

    # 공통 x축 (5% 마진)
    all_vals = pd.concat([s for _, s in datasets], ignore_index=True)
    x_min, x_max = float(all_vals.min()), float(all_vals.max())
    if np.isfinite(x_min) and np.isfinite(x_max) and x_min < x_max:
        pad = 0.05 * (x_max - x_min)
        x_min, x_max = x_min - pad, x_max + pad
    else:
        x_min, x_max = 0.0, 10.0
    x_grid = np.linspace(x_min, x_max, 512)

    # 2행×5열: 위 히스토그램(빈도), 아래 KDE(밀도)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), tight_layout=True)
    bins = 50

    # 위 행: 히스토그램
    for i, (name, s) in enumerate(datasets):
        ax = axes[0, i]
        ax.hist(s, bins=bins, edgecolor="black")
        ax.set_title(f"{name}\nHistogram")
        ax.set_xlabel("pIC50")
        ax.set_ylabel("Frequency")

    # 아래 행: KDE 밀도
    for i, (name, s) in enumerate(datasets):
        ax = axes[1, i]
        y_vals = None
        # 샘플이 2개 이상이고 분산이 0이 아니면 KDE
        if len(s) >= 2 and float(np.std(s)) > 0:
            try:
                kde = gaussian_kde(s.values)  # Scott's rule by default
                y_vals = kde.evaluate(x_grid)
            except Exception:
                y_vals = None
        if y_vals is not None and np.all(np.isfinite(y_vals)):
            ax.plot(x_grid, y_vals)
            ax.set_ylim(bottom=0)
        else:
            # KDE 불가 시, 정규화 히스토그램으로 대체 (밀도)
            counts, edges = np.histogram(s, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, counts, drawstyle="steps-mid")
            ax.set_ylim(bottom=0)
        ax.set_title(f"{name}\nKDE (Density)")
        ax.set_xlabel("pIC50")
        ax.set_ylabel("Density")

    out_path = result_dir / "pIC50_hist_and_kde_10panels.png"
    fig.savefig(out_path, dpi=150)
    # 아래 행: KDE 밀도
    for i, (name, s) in enumerate(datasets):
        ax = axes[1, i]
        y_vals = None
        # 샘플이 2개 이상이고 분산이 0이 아니면 KDE
        if len(s) >= 2 and float(np.std(s)) > 0:
            try:
                kde = gaussian_kde(s.values)  # Scott's rule by default
                y_vals = kde.evaluate(x_grid)
            except Exception:
                y_vals = None
        if y_vals is not None and np.all(np.isfinite(y_vals)):
            ax.plot(x_grid, y_vals)
            ax.set_ylim(bottom=0)
        else:
            # KDE 불가 시, 정규화 히스토그램으로 대체 (밀도)
            counts, edges = np.histogram(s, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, counts, drawstyle="steps-mid")
            ax.set_ylim(bottom=0)
        ax.set_title(f"{name}\nKDE (Density)")
        ax.set_xlabel("pIC50")
        ax.set_ylabel("Density")

    out_path = result_dir / "pIC50_hist_and_kde_10panels.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()
