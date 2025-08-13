import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re, unicodedata

# ── 설정 ─────────────────────────────────────────────────────────────
DATA_PATH = Path("/home1/won0316/DACON/JUMP_AI_2025_EST/data/CAS_KPBMA_MAP3K5_IC50s.xlsx")
SHEET = "MAP3K5 Ligand IC50s"   # ← 지정된 시트
VALUE_PREF = ["pX Value", "Single Value (Parsed)"]
TOP_N = 10

# ── 유틸: 컬럼 정규화 ────────────────────────────────────────────────
def normalize(txt):
    txt = unicodedata.normalize("NFKD", str(txt))
    return re.sub(r"[\W_]+", "", txt).lower()

# ── 1) 시트 로드 ─────────────────────────────────────────────────────
df = pd.read_excel(DATA_PATH, sheet_name=SHEET, header=1)
df.columns = df.columns.str.strip()

# ── 2) 핵심 컬럼 매핑 ────────────────────────────────────────────────
norm_map = {normalize(c): c for c in df.columns}
if "assayparameter" in norm_map:
    df.rename(columns={norm_map["assayparameter"]: "Assay Parameter"}, inplace=True)
if "species" in norm_map:
    df.rename(columns={norm_map["species"]: "Species"}, inplace=True)
if "disease" in norm_map:
    df.rename(columns={norm_map["disease"]: "Disease"}, inplace=True)

# ── 3) IC50 행 선택 ─────────────────────────────────────────────────
df_ic50 = df[df["Assay Parameter"].astype(str).str.upper() == "IC50"].copy()
if df_ic50.empty:
    raise RuntimeError("IC50 행을 찾지 못함 — Assay Parameter 값을 확인")

# ── 4) 값 컬럼 선택 ─────────────────────────────────────────────────
for c in VALUE_PREF:
    if c in df_ic50.columns:
        VALUE_COL = c
        break
else:
    raise RuntimeError("IC50 값을 나타내는 컬럼이 없습니다.")

# ── 5) 분포 그래프 ─────────────────────────────────────────────────
def plot_dist(group_col):
    cats = df_ic50[group_col].value_counts().nlargest(TOP_N).index
    sub = df_ic50[df_ic50[group_col].isin(cats)]
    sns.displot(sub, x=VALUE_COL, col=group_col, col_wrap=3,
                kde=True, facet_kws=dict(sharey=False), height=3)
    plt.tight_layout()
    plt.savefig(f"{group_col}_IC50_hist.png", dpi=300)
    plt.close()

for col in ["Assay Name", "Species", "Disease"]:
    if col in df_ic50.columns:
        plot_dist(col)

print("완료: *_IC50_hist.png 그래프가 생성되었습니다.")
