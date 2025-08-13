"""IC50 보간 처리 모듈."""

from __future__ import annotations

import logging
import random
import sys                     # ← 누락돼 있던 sys 추가
from pathlib import Path
from typing import Iterable
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdRGroupDecomposition as rdRGD
from tqdm.auto import tqdm

# 프로젝트 루트 패스 추가 --------------------------------------------------------
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# pyarrow 우선, 실패 시 fastparquet ---------------------------------------------
try:
    import pyarrow as _pa  # noqa: F401
    PARQUET_ENGINE = "pyarrow"
except Exception:  # pragma: no cover
    PARQUET_ENGINE = "fastparquet"

from utils.config_loader import CFG  # noqa: E402 (프로젝트 import는 뒤에)

# ------------------------------------------------------------------------------#
# 1. 시드 고정 함수
# ------------------------------------------------------------------------------#
def _set_seed(seed: int = 42) -> None:
    """재현성을 위한 RNG 시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    # RDKit 빌드마다 시드 함수가 다르므로 존재 여부 체크
    if hasattr(rdBase, "RandomSeed"):
        rdBase.RandomSeed(seed)
    elif hasattr(rdBase, "InitRandomSeed"):
        rdBase.InitRandomSeed(seed)
    elif hasattr(rdBase, "initRandomSeed"):
        rdBase.initRandomSeed(seed)
    else:
        logging.info("RDKit 빌드에 전역 시드 API가 없어 패스")


# ------------------------------------------------------------------------------#
# 2. 보간 파라미터
# ------------------------------------------------------------------------------#
MIN_NEIGHBORS = 5         # 유사 이웃 최소 개수
MAX_CV = 0.5              # 계수변동계수(Coef. of Variation) 임계값


def _low_uncertainty(values: Iterable[float]) -> bool:
    """표본 집합이 충분하고 변동성이 낮으면 True."""
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if len(vals) < MIN_NEIGHBORS:
        return False
    vals = np.array(vals, dtype=float)
    mean = vals.mean()
    if mean <= 0:
        return False
    cv = vals.std(ddof=0) / mean
    return cv < MAX_CV


# ------------------------------------------------------------------------------#
# 3. 데이터 로드 & 전처리
# ------------------------------------------------------------------------------#
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = Path(CFG.get_path("raw_dir"))
    cas = pd.read_excel(raw / "CAS_KPBMA_MAP3K5_IC50s.xlsx",
                        sheet_name="MAP3K5 Ligand IC50s")
    chembl = pd.read_csv(raw / "ChEMBL_ASK1(IC50).csv", sep=";")
    pubchem = pd.read_csv(raw / "Pubchem_ASK1.csv", low_memory=False)
    return cas, chembl, pubchem


def preprocess(
    cas: pd.DataFrame, chembl: pd.DataFrame, pubchem: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """세 파일을 공통 컬럼(SMILES·IC50_nM)으로 정규화."""

    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in df.columns}

        s_col = (cols.get("smiles") or cols.get("smile") or cols.get("smile(s)")
                 or next((c for c in df.columns if "smile" in c.lower()), df.columns[0]))
        v_col = (cols.get("activity_value(nm)") or cols.get("activity_value")
                 or cols.get("ic50_nm") or cols.get("standard value")
                 or next((c for c in df.columns if "value" in c.lower()), df.columns[1]))

        out = df[[s_col, v_col]].rename(columns={s_col: "SMILES", v_col: "IC50_nM"})
        out["IC50_nM"] = pd.to_numeric(out["IC50_nM"], errors="coerce")
        return out

    return _norm(cas), _norm(chembl), _norm(pubchem)


# ------------------------------------------------------------------------------#
# 4. Mol·R‑group 유틸
# ------------------------------------------------------------------------------#
def _get_mol(s: str | float) -> Chem.Mol | None:
    if not isinstance(s, str):
        return None
    s = s.strip()
    return Chem.MolFromSmiles(s) if s else None


def _extract_rgroups(mol: Chem.Mol) -> tuple[str, list[str]]:
    """Murcko scaffold + R‑groups 추출(SMILES 반환)."""
    if mol is None:
        return "", []

    core = MurckoScaffold.GetScaffoldForMol(mol)
    core_smiles = Chem.MolToSmiles(core, isomericSmiles=True)

    # RDKit 2023.09+ 시그니처: (cores, params)
    params = rdRGD.RGroupDecompositionParameters()
    params.symmetrize = False
    params.removeHydrogensPostMatch = True
    params.asSmiles = True

    rgd = rdRGD.RGroupDecomposition([core], params)
    rgd.Add(mol)
    rgd.Process()

    rows = rgd.GetRGroupsAsRows(asSmiles=True)
    rgroups: list[str] = []
    if rows:
        for k, v in rows[0].items():
            if k != "Core" and v:
                rgroups.append(v)
    return core_smiles, rgroups


def _hash_key(core: str, rgs: Iterable[str]) -> str:
    return f"{core}|{','.join(sorted(rgs))}"


# ------------------------------------------------------------------------------#
# 5. 결측 IC50 보간
# ------------------------------------------------------------------------------#
def impute_missing(
    cas: pd.DataFrame, chembl: pd.DataFrame, pubchem: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """PubChem 결측 IC50을 스캐폴드·유사도 기반으로 보간."""
    _set_seed(42)

    df = pd.concat(
        [cas.assign(_src="cas"),
         chembl.assign(_src="chembl"),
         pubchem.assign(_src="pubchem")],
        ignore_index=True,
    )

    df["mol"] = df["SMILES"].apply(_get_mol)
    df["scaffold"], df["rgroups"] = zip(*df["mol"].apply(_extract_rgroups))
    df["hash_key"] = [_hash_key(c, r) for c, r in zip(df["scaffold"], df["rgroups"],
                                                      strict=True)]
    df["fp"] = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048,
                                                      useChirality=True) if m else None
                for m in df["mol"]]

    known = df[df["IC50_nM"].notna()]
    med_by_hash = known.groupby("hash_key")["IC50_nM"].median().to_dict()
    count_by_hash = known.groupby("hash_key")["IC50_nM"].count().to_dict()
    std_by_hash = known.groupby("hash_key")["IC50_nM"].std().to_dict()

    all_fps = list(known["fp"])
    all_idx = list(known.index)

    counts = {"a": 0, "b": 0, "c": 0}

    for idx, row in tqdm(
        df[df["IC50_nM"].isna()].iterrows(),
        total=df["IC50_nM"].isna().sum(),
        disable=True,
    ):
        val = med_by_hash.get(row["hash_key"])
        method = None

        # ── 단계 a: 동일 스캐폴드+R‑group 통계 ───────────────────────
        if val is not None:
            cnt = count_by_hash.get(row["hash_key"], 0)
            std = std_by_hash.get(row["hash_key"], float("inf"))
            if cnt >= MIN_NEIGHBORS and (std / max(val, 1e-8)) < MAX_CV:
                method = "a"
            else:
                val = None

        # ── 단계 b: 동일 스캐폴드, Tanimoto ≥0.7 ────────────────────
        if val is None and row["fp"] is not None:
            same = known[known["scaffold"] == row["scaffold"]]
            sims = DataStructs.BulkTanimotoSimilarity(row["fp"], list(same["fp"]))
            pairs = [(v, s) for v, s in zip(same["IC50_nM"], sims, strict=True) if s >= 0.7]
            if pairs:
                top = [v for v, _ in sorted(pairs, key=lambda x: -x[1])[:MIN_NEIGHBORS]]
                if _low_uncertainty(top):
                    val = float(np.mean(top))
                    method = "b"

        # ── 단계 c: 전체 데이터, Tanimoto ≥0.5 ─────────────────────
        if val is None and row["fp"] is not None:
            sims = DataStructs.BulkTanimotoSimilarity(row["fp"], all_fps)
            pairs = [(known.loc[i, "IC50_nM"], s)
                     for i, s in zip(all_idx, sims, strict=True) if s >= 0.5]
            if pairs:
                top = [v for v, _ in sorted(pairs, key=lambda x: -x[1])[:MIN_NEIGHBORS]]
                if _low_uncertainty(top):
                    val = float(np.mean(top))
                    method = "c"

        # ── 결과 반영 ───────────────────────────────────────────────
        if val is not None:
            df.at[idx, "IC50_nM"] = val
            counts[method] += 1

    logging.info("총 변환 %d건 (a:%d, b:%d, c:%d)",
                 sum(counts.values()), counts["a"], counts["b"], counts["c"])
    return df, counts


# ------------------------------------------------------------------------------#
# 6. 결과 저장
# ------------------------------------------------------------------------------#
def save_outputs(
    df: pd.DataFrame,
    counts: dict[str, int],
    pubchem_name: str = "Pubchem_ASK1_imputed.csv",
    all_name: str = "all_datasets_imputed.parquet",
) -> None:
    raw = Path(CFG.get_path("raw_dir"))

    df[df["_src"] == "pubchem"][["SMILES", "IC50_nM"]].to_csv(
        raw / pubchem_name, index=False
    )
    df[["SMILES", "IC50_nM"]].to_parquet(
        raw / all_name, index=False, engine=PARQUET_ENGINE
    )
    print(f"총 변환 {sum(counts.values())}건  a:{counts['a']}  b:{counts['b']}  c:{counts['c']}")
