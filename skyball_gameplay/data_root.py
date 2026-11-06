"""Locate SkyBall data: relative dataset paths resolve under the unified_dataset root (SKYBALL_DATA_ROOT)."""
import os
from pathlib import Path


DEFAULT_DATA_ROOT = "/mnt/t/data/vball/skyball/unified_dataset"


def data_root() -> Path:
    return Path(os.environ.get("SKYBALL_DATA_ROOT", DEFAULT_DATA_ROOT))


def dataset_path(path) -> Path:
    """A relative path is taken under the data root; an absolute path is returned as is."""
    path = Path(path)
    return path if path.is_absolute() else data_root() / path
