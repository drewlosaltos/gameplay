"""Checkpoint-compatible SkyBall gameplay model construction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import timm
import torch
import torchvision.models


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TemporalAveragePoolClassifier(torch.nn.Module):
    """Average ResNet image embeddings over a fixed temporal window."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not hasattr(backbone, "fc"):
            raise ValueError("Temporal classifier requires a backbone with an fc layer")
        feature_dim = int(backbone.fc.in_features)
        self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.classifier = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = inputs.shape
        flattened = inputs.reshape(batch_size * num_frames, channels, height, width)
        embeddings = torch.flatten(self.encoder(flattened), 1)
        embeddings = embeddings.reshape(batch_size, num_frames, -1).mean(dim=1)
        return self.classifier(self.dropout(embeddings))


class TemporalFeatureAveragePoolClassifier(torch.nn.Module):
    """Average generic encoder embeddings over a fixed temporal window."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.classifier = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = inputs.shape
        flattened = inputs.reshape(batch_size * num_frames, channels, height, width)
        embeddings = self.encoder(flattened)
        if embeddings.ndim > 2:
            embeddings = torch.flatten(embeddings, 1)
        embeddings = embeddings.reshape(batch_size, num_frames, -1).mean(dim=1)
        return self.classifier(self.dropout(embeddings))


def normalization_for_backbone(
    timm_model: str = "",
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the pretrained input normalization for an encoder."""
    if not timm_model:
        return IMAGENET_MEAN, IMAGENET_STD
    config = timm.get_pretrained_cfg(timm_model)
    return (
        tuple(float(value) for value in config.mean),
        tuple(float(value) for value in config.std),
    )


def replace_classifier(model: torch.nn.Module, num_classes: int) -> None:
    """Replace a torchvision-style classification layer in place."""
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, torch.nn.Linear):
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
            return
        if isinstance(model.classifier, torch.nn.Sequential):
            for index in range(len(model.classifier) - 1, -1, -1):
                layer = model.classifier[index]
                if isinstance(layer, torch.nn.Linear):
                    model.classifier[index] = torch.nn.Linear(
                        layer.in_features,
                        num_classes,
                    )
                    return
    raise ValueError(f"Cannot replace classifier for {type(model).__name__}")


def create_gameplay_model(
    *,
    arch: str,
    timm_model: str,
    input_frames: int,
    num_classes: int,
    temporal_dropout: float = 0.0,
    pretrained: bool = False,
) -> torch.nn.Module:
    """Create the model layout used by historical gameplay checkpoints."""
    if timm_model:
        if input_frames != 4:
            raise ValueError("timm gameplay models require four-frame input")
        encoder = timm.create_model(
            timm_model,
            pretrained=pretrained,
            num_classes=0,
        )
        return TemporalFeatureAveragePoolClassifier(
            encoder,
            feature_dim=int(encoder.num_features),
            num_classes=num_classes,
            dropout=temporal_dropout,
        )
    if arch not in torchvision.models.__dict__:
        raise ValueError(f"Unsupported torchvision architecture: {arch}")
    constructor = torchvision.models.__dict__[arch]
    backbone = constructor(pretrained=True) if pretrained else constructor(weights=None)
    if input_frames == 4:
        return TemporalAveragePoolClassifier(
            backbone,
            num_classes,
            dropout=temporal_dropout,
        )
    replace_classifier(backbone, num_classes)
    return backbone


def strip_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove DataParallel/DDP prefixes from checkpoint keys."""
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def checkpoint_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Normalize model metadata stored by legacy and current trainers."""
    arch = str(checkpoint.get("arch", "resnet50"))
    timm_model = str(checkpoint.get("timm_model") or "")
    classes = list(checkpoint.get("classes") or ["non_gameplay", "gameplay"])
    if "gameplay" not in classes:
        raise ValueError(f"Checkpoint classes do not include gameplay: {classes}")
    input_frames = int(checkpoint.get("input_frames", 1))
    mean = checkpoint.get("normalization_mean")
    std = checkpoint.get("normalization_std")
    if mean is None or std is None:
        mean, std = normalization_for_backbone(timm_model)
    return {
        "arch": arch,
        "timm_model": timm_model,
        "input_frames": input_frames,
        "label_mode": str(checkpoint.get("label_mode", "multiclass")),
        "classes": classes,
        "num_classes": int(checkpoint.get("num_classes", len(classes))),
        "gameplay_class_index": classes.index("gameplay"),
        "image_size": list(checkpoint.get("image_size") or [512, 512]),
        "normalization_mean": list(mean),
        "normalization_std": list(std),
        "temporal_dropout": float(checkpoint.get("temporal_dropout", 0.0)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
    }


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device | str,
    *,
    require_input_frames: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstruct a gameplay model and load a legacy-compatible checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint_metadata(checkpoint)
    if (
        require_input_frames is not None
        and metadata["input_frames"] != require_input_frames
    ):
        raise ValueError(
            f"Expected a {require_input_frames}-frame checkpoint, "
            f"got {metadata['input_frames']}"
        )
    model = create_gameplay_model(
        arch=metadata["arch"],
        timm_model=metadata["timm_model"],
        input_frames=metadata["input_frames"],
        num_classes=metadata["num_classes"],
        temporal_dropout=metadata["temporal_dropout"],
        pretrained=False,
    )
    model.load_state_dict(strip_module_prefix(checkpoint["state_dict"]))
    model.to(device)
    model.eval()
    return model, metadata


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
