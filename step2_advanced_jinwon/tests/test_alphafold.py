import unittest

try:
    from src.featurization.structure import add_structure_features
    import pandas as pd
    DEP_AVAILABLE = True
except Exception:
    DEP_AVAILABLE = False


class TestAlphafoldFeature(unittest.TestCase):
    @unittest.skipUnless(DEP_AVAILABLE, 'alphafold dependencies not installed')
    def test_add_structure_features(self):
        df = pd.DataFrame({'canonical_smiles': ['CCO']})
        out = add_structure_features(df)
        self.assertIn('af_rmsd', out.columns)


if __name__ == '__main__':
    unittest.main()