"""Optional, checkpoint-compatible acceleration for Qwen gameplay training."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


MODES = {"none", "compile_bf16", "compile_fp8"}


def prepare_training_acceleration(model: nn.Module, mode: str) -> dict[str, Any]:
    """Convert encoder linear compute before DDP without changing parameters or keys."""
    if mode not in MODES:
        raise ValueError(f"Unknown training acceleration mode: {mode}")
    report: dict[str, Any] = {"mode": mode, "fp8_linear_count": 0}
    if mode != "compile_fp8":
        return report

    from torchao.float8 import Float8LinearConfig
    from torchao.float8.float8_linear import Float8Linear

    class TrainingFloat8Linear(Float8Linear):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            if not self.training:
                return F.linear(inputs, self.weight, self.bias)
            return super().forward(inputs)

    encoder = model.encoder
    config = Float8LinearConfig.from_recipe_name("tensorwise")
    parameters_before = dict(encoder.named_parameters())
    keys_before = list(encoder.state_dict())

    def convert(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if type(child) is nn.Linear:
                replacement = TrainingFloat8Linear.from_float(child, config=config)
                replacement.train(child.training)
                setattr(parent, name, replacement)
                report["fp8_linear_count"] += 1
            else:
                convert(child)

    convert(encoder)
    if keys_before != list(encoder.state_dict()):
        raise RuntimeError("FP8 conversion changed checkpoint keys")
    if any(parameters_before[name] is not value for name, value in encoder.named_parameters()):
        raise RuntimeError("FP8 conversion replaced learned parameters")
    return report


def install_training_compiler(model: nn.Module, mode: str, compile_mode: str) -> None:
    """Compile training forward only; keep normal evaluation and checkpoint layout."""
    if mode not in MODES:
        raise ValueError(f"Unknown training acceleration mode: {mode}")
    if mode == "none":
        return
    original_forward = model.forward
    compiled_forward = torch.compile(
        original_forward, fullgraph=False, dynamic=False, mode=compile_mode,
    )

    def forward(*args: Any, **kwargs: Any) -> Any:
        if model.training:
            return compiled_forward(*args, **kwargs)
        return original_forward(*args, **kwargs)

    model.forward = forward
