"""Standalone Qwen vision tower for experimental gameplay classification."""

from __future__ import annotations

import json
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeVisionConfig,
    Qwen3_5MoeVisionModel,
    apply_rotary_pos_emb_vision,
)


def patchify(images: torch.Tensor, patch: int = 16, merge: int = 2) -> torch.Tensor:
    """Match Qwen's single-image temporal duplication and spatial merge order."""
    batch, channels, height, width = images.shape
    gh, gw = height // patch, width // patch
    if height % (patch * merge) or width % (patch * merge):
        raise ValueError("Image dimensions must be multiples of patch * merge")
    patches = images.reshape(batch, channels, gh // merge, merge, patch, gw // merge, merge, patch)
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    return patches.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, 2, -1, -1).reshape(
        batch * gh * gw, channels * 2 * patch * patch
    )


def uniform_attention(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
    """Batch equal-sized crops without the reference implementation's per-image loop."""
    tokens = self.jersey_tokens_per_image
    length = hidden_states.shape[0]
    batch = length // tokens
    query, key, value = self.qkv(hidden_states).reshape(length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    query, key = apply_rotary_pos_emb_vision(query, key, *position_embeddings)
    def batched(tensor):
        return tensor.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
    output = F.scaled_dot_product_attention(
        batched(query), batched(key), batched(value), dropout_p=0.0, scale=self.scaling
    )
    return self.proj(output.transpose(1, 2).reshape(length, self.dim))


def image_hw(image_size):
    """Accept a legacy square size or an explicit (height, width)."""
    height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError('Image height and width must be positive multiples of 32')
    return height, width


def load_vision(snapshot: Path, image_size, fast_attention: bool = True):
    saved = json.loads((snapshot / "config.json").read_text())
    config = Qwen3_5MoeVisionConfig(**saved["vision_config"])
    config._attn_implementation = "sdpa"
    vision = Qwen3_5MoeVisionModel(config)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.visual."
    keys = {key: filename for key, filename in index.items() if key.startswith(prefix)}
    weights = {}
    for filename in sorted(set(keys.values())):
        with safe_open(snapshot / filename, framework="pt", device="cpu") as tensors:
            for key, shard in keys.items():
                if shard == filename:
                    weights[key.removeprefix(prefix)] = tensors.get_tensor(key)
    vision.load_state_dict(weights, strict=True)
    if fast_attention:
        height, width = image_hw(image_size)
        for block in vision.blocks:
            block.attn.jersey_tokens_per_image = (height // config.patch_size) * (width // config.patch_size)
            block.attn.forward = types.MethodType(uniform_attention, block.attn)
    return vision


class QwenFrameEncoder(nn.Module):
    """Spatially average merged visual tokens; temporal averaging stays external."""

    def __init__(self, snapshot: str | Path, image_size=(512, 512)):
        super().__init__()
        self.image_size = image_hw(image_size)
        self.vision = load_vision(Path(snapshot), image_size)
        self.num_features = 2048

    def forward(self, images):
        height, width = self.image_size
        if tuple(images.shape[-2:]) != (height, width):
            raise ValueError('Unexpected Qwen image dimensions')
        batch = images.shape[0]
        grid = torch.tensor([[1, height // 16, width // 16]] * batch,
                            dtype=torch.long, device=images.device)
        tokens = self.vision(patchify(images), grid).pooler_output
        return tokens.reshape(batch, -1, self.num_features).mean(dim=1)
