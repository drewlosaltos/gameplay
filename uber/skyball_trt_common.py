#!/usr/bin/env python3
"""Shared helpers for SkyBall RF-DETR TensorRT experiments."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn


DETECTOR_REPO = Path("/home/atao/vsdevel/skyball/rf-detr-mf-skyball")
DEFAULT_CHECKPOINT = Path(
    "/mnt/t/output/rf-detr-mf/runs/skyball/"
    "large_dense_v4_addcats_sparse_v3_2g/checkpoint_best_ema.pth"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/t/output/rf-detr-mf/benchmarks/"
    "large_dense_v4_addcats_sparse_v3_2g_trt"
)


def install_detector_repo() -> None:
    """Make the local detector source importable."""
    src = str(DETECTOR_REPO / "src")
    repo = str(DETECTOR_REPO)
    if src not in sys.path:
        sys.path.insert(0, src)
    if repo not in sys.path:
        sys.path.insert(0, repo)


install_detector_repo()

from rfdetr.util.misc import NestedTensor  # noqa: E402
from rfdetr.util.tracking_model_loader import load_model_from_checkpoint  # noqa: E402


class CenterDetectorExportWrapper(nn.Module):
    """Export wrapper that preserves the normal dense-window inference path."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return center-frame boxes/logits from ``frames`` shaped ``[B,T,H,W]``."""
        batch, _, height, width = frames.shape
        mask = torch.zeros(
            (batch, height, width),
            dtype=torch.bool,
            device=frames.device,
        )
        outputs = self.model(NestedTensor(frames, mask))
        return outputs["pred_boxes"], outputs["pred_logits"]


FULL_OUTPUT_NAMES = (
    "pred_boxes",
    "pred_logits",
    "pred_boxes_all_frames",
    "pred_logits_all_frames",
    "pred_frame_event_logits",
    "pred_frame_skill_logits",
    "pred_frame_side_logits",
    "pred_frame_event_occ_logits",
    "pred_frame_actor_logits",
    "pred_side_logits",
    "pred_jump_logits_all_frames",
    "pred_query_features",
)


class FullDetectorExportWrapper(nn.Module):
    """Export wrapper for full multi-frame tracking/video outputs."""

    output_names = FULL_OUTPUT_NAMES

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Return all tensors required by tracking visualization/evaluation."""
        batch, _, height, width = frames.shape
        mask = torch.zeros(
            (batch, height, width),
            dtype=torch.bool,
            device=frames.device,
        )
        outputs = self.model(NestedTensor(frames, mask))
        return tuple(outputs[name] for name in self.output_names)


def load_export_model(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    device: str | torch.device = "cuda",
    checkpoint_state: str = "model",
    output_mode: str = "center",
) -> Tuple[CenterDetectorExportWrapper, Any]:
    """Load the large SkyBall checkpoint and wrap it for export/benchmarking."""
    torch_device = torch.device(device)
    model, saved_args = load_model_from_checkpoint(
        str(checkpoint),
        torch_device,
        checkpoint_state=checkpoint_state,
    )
    model.eval()
    if output_mode == "center":
        wrapper = CenterDetectorExportWrapper(model)
    elif output_mode == "all-frames":
        wrapper = FullDetectorExportWrapper(model)
    else:
        raise ValueError(f"Unsupported output_mode: {output_mode}")
    wrapper = wrapper.to(torch_device).eval()
    return wrapper, saved_args


def make_input(
    batch_size: int,
    num_frames: int,
    resolution: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a normalized synthetic dense-window input."""
    return torch.randn(
        batch_size,
        num_frames,
        resolution,
        resolution,
        device=device,
        dtype=dtype,
    )


def precision_to_dtype(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def autocast_context(device: torch.device, precision: str):
    """Return an autocast context for PyTorch benchmarks."""
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(
        device_type="cuda",
        dtype=precision_to_dtype(precision),
        enabled=True,
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def default_artifact_dir() -> Path:
    return Path(os.environ.get("SKYBALL_TRT_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
