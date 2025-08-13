import unittest

try:
    import pandas as pd
    from src.featurization.complex_features import add_complex_features
    from Bio.PDB import PDBParser  # noqa: F401
    DEP_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    DEP_AVAILABLE = False


class TestComplexFeatures(unittest.TestCase):
    @unittest.skipUnless(DEP_AVAILABLE, "biopython not installed")
    def test_add_complex_features(self):
        pdb_content = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            "ATOM      2  O   ALA A   1       0.000   0.000   1.20  1.00 20.00           O\n"
            "TER\n"
            "ATOM      3  N   ALA B   1       0.000   0.000   2.80  1.00 20.00           N\n"
            "ATOM      4  O   ALA B   1       0.000   0.000   4.00  1.00 20.00           O\n"
            "TER\nEND\n"
        )
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            pdb_dir = Path(td)
            (pdb_dir / "6EJL.pdb").write_text(pdb_content)
            df = pd.DataFrame({"canonical_smiles": ["CCO"]})
            out = add_complex_features(df, pdb_dir=pdb_dir)
            for col in [
                "complex_interaction_residues",
                "complex_hbond_count",
                "complex_interface_area",
            ]:
                self.assertIn(col, out.columns)
                self.assertTrue(out[col].notna().all())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

