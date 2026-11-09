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


def hf_hub_cache() -> Path:
    """The Hugging Face hub cache: HF_HUB_CACHE, else HF_HOME/hub, else ~/.cache/huggingface/hub."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"


def hf_snapshot_path(repo_id: str, revision: str) -> Path:
    """Where `hf download <repo_id> --revision <revision>` puts the snapshot."""
    return hf_hub_cache() / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision


def qwen_snapshot(stored: str | Path) -> Path:
    """Resolve a Qwen snapshot path recorded in a config or checkpoint on this machine.

    SKYBALL_QWEN_SNAPSHOT overrides it. Otherwise an existing path is kept, and a missing
    path inside some hub cache (``.../hub/models--X/snapshots/<rev>``) is re-rooted under
    this machine's hub cache.
    """
    if os.environ.get("SKYBALL_QWEN_SNAPSHOT"):
        return Path(os.environ["SKYBALL_QWEN_SNAPSHOT"])
    path = Path(stored)
    if path.exists():
        return path
    parts = path.parts
    for index in range(len(parts) - 2):
        if parts[index].startswith("models--") and parts[index + 1] == "snapshots":
            return hf_hub_cache().joinpath(*parts[index:])
    return path
