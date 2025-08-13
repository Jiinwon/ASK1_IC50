import unittest

try:
    import pandas as pd
    from src.featurization.ecfp import featurize
    DEP_AVAILABLE = True
except Exception:
    DEP_AVAILABLE = False


class TestFeaturization(unittest.TestCase):
    @unittest.skipUnless(DEP_AVAILABLE, 'dependencies not installed')
    def test_ecfp(self):
        df = pd.DataFrame({'canonical_smiles': ['CCO']})
        out = featurize(df)
        self.assertIn(0, out.columns)


if __name__ == '__main__':
    unittest.main()
