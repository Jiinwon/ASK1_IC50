"""ECFP featurisation utilities."""

from __future__ import annotations

import pandas as pd
from tqdm.auto import tqdm

from utils.chem_utils import compute_ecfp
from utils.config_loader import CFG

tqdm.pandas(desc="ecfp")


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Append ECFP features to ``df`` using parameters from :mod:`CFG`.

    The function honours two configuration flags:

    ``ecfp_use_chirality`` (bool)
        Whether to encode chiral information in the fingerprint.
    ``use_feature_morgan`` (bool)
        If ``True`` the RDKit "feature Morgan" variant is used.  Otherwise a
        standard ECFP is generated.
    """

    bits = CFG.get_hparam("ecfp_bits")
    radius = CFG.get_hparam("ecfp_radius")
    use_chirality = CFG.get_hparam("ecfp_use_chirality")
    if use_chirality is None:
        use_chirality = True
    use_feature = CFG.get_hparam("use_feature_morgan")
    if use_feature is None:
        use_feature = False

    if use_feature:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        def _feature_fp(smiles: str):
            mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
            if mol is None:
                return [0] * bits
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius,
                nBits=bits,
                useChirality=use_chirality,
                useFeatures=True,
            )
            return list(fp)

        fps = df["canonical_smiles"].progress_apply(_feature_fp)
    else:
        fps = df["canonical_smiles"].progress_apply(
            lambda s: compute_ecfp(s, radius, bits, use_chirality=use_chirality)
        )

    fp_df = pd.DataFrame(fps.tolist(), index=df.index)
    return pd.concat([df, fp_df], axis=1)
