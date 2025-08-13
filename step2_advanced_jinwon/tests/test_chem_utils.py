import unittest

try:
    from utils.chem_utils import standardize_smiles
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False


class TestChemUtils(unittest.TestCase):
    @unittest.skipUnless(RDKit_AVAILABLE, 'rdkit not installed')
    def test_standardize(self):
        smi = 'CC(=O)O'
        self.assertIsInstance(standardize_smiles(smi), str)


if __name__ == '__main__':
    unittest.main()
