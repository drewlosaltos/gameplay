"""Regression tests for temporal SigLIP gameplay training support."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from skyball_gameplay import data as gameplay_data
from skyball_gameplay import evaluation
from skyball_gameplay import modeling
from skyball_gameplay import training


class MeanFeatureEncoder(nn.Module):
    """Return one spatial mean feature per input channel."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into a three-value feature vector."""
        return images.mean(dim=(2, 3))


def test_temporal_feature_classifier_averages_encoder_features() -> None:
    """The generic temporal head should average one feature vector per frame."""
    model = modeling.TemporalFeatureAveragePoolClassifier(
        MeanFeatureEncoder(),
        feature_dim=3,
        num_classes=2,
    )
    with torch.no_grad():
        model.classifier.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        model.classifier.bias.zero_()
    images = torch.zeros(1, 4, 3, 2, 2)
    images[:, 0, 0] = 2.0
    images[:, 1, 0] = 2.0
    images[:, 2, 1] = 4.0
    images[:, 3, 1] = 4.0

    output = model(images)

    assert torch.equal(output, torch.tensor([[1.0, 2.0]]))


def test_optimizer_uses_conservative_backbone_learning_rate() -> None:
    """AdamW should preserve the jersey-derived 0.05 encoder LR multiplier."""
    model = modeling.TemporalFeatureAveragePoolClassifier(
        MeanFeatureEncoder(),
        feature_dim=3,
        num_classes=2,
    )

    optimizer = training.build_optimizer(
        model,
        learning_rate=1e-4,
        weight_decay=1e-4,
        backbone_lr_multiplier=0.05,
        optimizer_name="adamw",
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert [group["lr"] for group in optimizer.param_groups] == [5e-6, 1e-4]


def test_siglip_uses_pretrained_normalization() -> None:
    """SigLIP input normalization should come from its timm pretrained config."""
    mean, std = modeling.normalization_for_backbone(
        "vit_base_patch16_siglip_224.v2_webli"
    )

    assert mean == (0.5, 0.5, 0.5)
    assert std == (0.5, 0.5, 0.5)


def test_temporal_dataset_returns_each_requested_frame_once(tmp_path: Path) -> None:
    """A four-frame manifest must not be expanded to eight duplicate tensors."""
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "class_to_idx.json").write_text(
        json.dumps({"non_gameplay": 0, "gameplay": 1}),
        encoding="utf-8",
    )
    frame_paths = []
    for index in range(4):
        relative_path = Path("train") / "gameplay" / f"frame_{index}.jpg"
        image_path = tmp_path / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(index, index, index)).save(image_path)
        frame_paths.append(relative_path.as_posix())
    record = {
        "split": "trn",
        "frame_paths": frame_paths,
        "class_name": "gameplay",
        "binary_label": 1,
    }
    (metadata / "windows.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
    dataset = gameplay_data.TemporalWindowDataset(
        str(tmp_path),
        "trn",
        transform=transforms.ToTensor(),
        input_frames=4,
        label_mode="binary",
    )

    images, target = dataset[0]

    assert images.shape == (4, 3, 8, 8)
    assert target == 1


def test_gradient_accumulation_steps_on_boundaries_and_final_batch() -> None:
    """Accumulation should step on full groups and flush a final partial group."""
    decisions = [
        training.should_optimizer_step(index, batch_count=5, grad_accum_steps=2)
        for index in range(5)
    ]
    divisors = [
        training.gradient_accumulation_divisor(index, batch_count=5, grad_accum_steps=2)
        for index in range(5)
    ]

    assert decisions == [False, True, False, True, True]
    assert divisors == [2, 2, 2, 2, 1]


def test_evaluator_caps_large_temporal_batch_prefetch() -> None:
    """A 512-pixel temporal batch must not queue sixteen multi-GiB tensors."""
    workers = evaluation.bounded_eval_workers(
        requested_workers=16,
        batch_size=128,
        input_frames=4,
        image_size=[512, 512],
        prefetch_budget_gib=8.0,
    )

    assert workers == 5
