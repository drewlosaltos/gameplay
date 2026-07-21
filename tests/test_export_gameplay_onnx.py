"""Tests for the portable gameplay ONNX export contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "export_gameplay_onnx.py"
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("export_gameplay_onnx", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DummyTemporalModel(torch.nn.Module):
    """Return deterministic two-class logits for temporal inputs."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Use the input mean as the gameplay logit."""
        gameplay = inputs.mean(dim=(1, 2, 3, 4))
        return torch.stack((-gameplay, gameplay), dim=1)


def test_export_wrapper_returns_logits_and_gameplay_probability() -> None:
    """The runtime graph should expose both raw logits and the gate score."""
    wrapper = MODULE.GameplayExportWrapper(
        DummyTemporalModel(), gameplay_class_index=1
    )
    inputs = torch.stack(
        (
            torch.zeros((4, 3, 2, 2)),
            torch.ones((4, 3, 2, 2)),
        )
    )

    logits, gameplay_score = wrapper(inputs)

    assert logits.shape == (2, 2)
    assert gameplay_score.shape == (2,)
    torch.testing.assert_close(gameplay_score, logits.softmax(dim=1)[:, 1])


def test_runtime_contract_records_gate_preprocessing() -> None:
    """The sidecar contract should completely specify pipeline input semantics."""
    contract = MODULE.runtime_contract(
        model_metadata={
            "input_frames": 4,
            "image_size": [512, 512],
            "normalization_mean": [0.5, 0.5, 0.5],
            "normalization_std": [0.5, 0.5, 0.5],
            "classes": ["non_gameplay", "gameplay"],
            "gameplay_class_index": 1,
        },
        batch_size=1,
        dynamic_batch=True,
        opset=17,
    )

    assert contract["input"]["shape"] == ["batch", 4, 3, 512, 512]
    assert contract["input"]["color_mode"] == "RGB"
    assert contract["input"]["tensor_scale"] == "float_[0,1]_then_normalize"
    assert contract["outputs"]["gameplay_score"]["decision"] == (
        "gameplay when score >= gate threshold"
    )
    assert contract["temporal_sampling"]["frame_spacing_seconds"] == 1.0
