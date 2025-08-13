import yaml
from pathlib import Path

class Config:
    def __init__(self, paths_file='config/paths.yaml', hparams_file='config/hyperparams.yaml'):
        self.paths = yaml.safe_load(Path(paths_file).read_text())
        self.hparams = yaml.safe_load(Path(hparams_file).read_text())

    def get_path(self, key):
        return self.paths.get(key)

    def get_hparam(self, key):
        return self.hparams.get(key)

CFG = Config()
