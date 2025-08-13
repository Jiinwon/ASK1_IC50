import yaml
from pathlib import Path


class Config:
    """Load project configuration files.

    The paths are resolved relative to the project root so that the module can
    be imported from anywhere without relying on the current working directory.
    """

    def __init__(
        self,
        paths_file: str = "config/paths.yaml",
        hparams_file: str = "config/hyperparams.yaml",
    ):
        root = Path(__file__).resolve().parents[1]
        self.root = root
        self.paths = yaml.safe_load((root / paths_file).read_text())
        self.hparams = yaml.safe_load((root / hparams_file).read_text())

    def get_path(self, key: str) -> str | None:
        """Return an absolute path for ``key`` defined in ``paths.yaml``."""
        val = self.paths.get(key)
        if val is None:
            return None
        return str((self.root / val).resolve())

    def get_hparam(self, key):
        return self.hparams.get(key)

CFG = Config()
