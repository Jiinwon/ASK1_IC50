"""Utility functions for cheminformatics tasks.

This module centralises helpers used across the project.  The functions here
are intentionally lightweight so they can be reused by featurisation as well as
analysis code without pulling in heavy dependencies.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys

# Suppress RDKit warnings and parse errors from being printed to stderr.  The
# logs are still accessible programmatically if needed but won't clutter test
# output.
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

logger = logging.getLogger(__name__)

def standardize_smiles(smiles: str) -> str:
    """
    SMILES 문자열을 받아 RDKit Mol 객체로 파싱한 뒤,
    isomeric SMILES 형태의 canonical SMILES 문자열로 반환합니다.
    결측값이나 파싱 실패 시 빈 문자열을 반환합니다.
    """
    # 1) 결측값 처리: None, NaN, non-str
    if pd.isna(smiles) or not isinstance(smiles, str):
        return ''
    s = smiles.strip()
    if not s:
        return ''
    
    # 2) RDKit 파싱
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return ''
    
    # 3) canonical SMILES 반환
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def compute_ecfp(
    smiles: str,
    radius: int,
    n_bits: int,
    use_chirality: bool = False,
) -> List[int]:
    """Return the Extended Connectivity Fingerprint for ``smiles``.

    Parameters
    ----------
    smiles:
        Input SMILES string.  Non-string or empty values result in an all-zero
        fingerprint.
    radius:
        ECFP radius ("ECFP4" corresponds to ``radius=2``).
    n_bits:
        Number of bits in the fingerprint.
    use_chirality:
        Whether to encode chiral information via RDKit's ``useChirality`` flag.

    Returns
    -------
    list[int]
        The fingerprint as a Python list of ints.
    """

    # 1) 결측값 처리
    if pd.isna(smiles) or not isinstance(smiles, str):
        logger.debug("compute_ecfp: invalid smiles %r", smiles)
        return [0] * n_bits
    s = smiles.strip()
    if not s:
        logger.debug("compute_ecfp: empty smiles after strip")
        return [0] * n_bits

    # 2) RDKit 파싱 및 fingerprint 생성
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        logger.debug("compute_ecfp: RDKit failed to parse %s", s)
        return [0] * n_bits

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=n_bits, useChirality=use_chirality
    )
    logger.debug(
        "compute_ecfp: generated %d-bit fingerprint with chirality=%s", n_bits, use_chirality
    )
    return list(fp)


def compute_maccs(smiles: str) -> list[int]:
    """Return the MACCS key fingerprint for ``smiles`` as a list of ints."""
    n_bits = 167
    if pd.isna(smiles) or not isinstance(smiles, str):
        return [0] * n_bits
    s = smiles.strip()
    if not s:
        return [0] * n_bits

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return [0] * n_bits

    fp = MACCSkeys.GenMACCSKeys(mol)
    return list(fp)
