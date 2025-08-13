import unittest

try:
    import pandas as pd
    from src.models.ensemble import EnsembleModel
    DEP_AVAILABLE = True
except Exception:
    DEP_AVAILABLE = False


class TestEnsembleModel(unittest.TestCase):
    @unittest.skipUnless(DEP_AVAILABLE, 'dependencies not installed')
    def test_predict_shape(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]})
        y = pd.Series([1.0, 2.0, 3.0])
        model = EnsembleModel({}, n_models=2)
        model.train(X, y)
        mean, std = model.predict(X, return_std=True)
        self.assertEqual(mean.shape[0], 3)
        self.assertEqual(std.shape[0], 3)


if __name__ == '__main__':
    unittest.main()
