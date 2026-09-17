"""Protect full-court geometry and distributed checkpoint-selection metrics."""

import pytest
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from skyball_gameplay.data import build_transforms
from skyball_gameplay.training import build_parser, metrics_from_confusion, validate


def test_full_frame_keeps_both_edges_and_matches_validation_geometry():
    args = build_parser().parse_args(["--train-geometry", "full_frame", "--image-size", "32", "32"])
    args.normalization_mean = (0, 0, 0)
    args.normalization_std = (1, 1, 1)
    image = Image.new("RGB", (160, 90), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 20, 89), fill="red")
    draw.rectangle((139, 0, 159, 89), fill="blue")
    train, val = build_transforms(args)
    actual, expected = train(image), val(image)
    assert torch.equal(actual, expected) or torch.equal(actual, expected.flip(-1))
    assert actual[0].max() == 1 and actual[2].max() == 1


def test_binary_metric_handles_empty_and_no_predicted_positives():
    assert metrics_from_confusion(torch.zeros(2, 2, dtype=torch.long))["f1"] == 0
    metrics = metrics_from_confusion(torch.tensor([[4, 0], [6, 0]]))
    assert metrics["accuracy"] == .4
    assert metrics["f1"] == 0
    assert metrics["balanced_accuracy"] == .5


def test_distributed_validation_counts_tail_once(monkeypatch):
    # Four prefix samples are distributed; the fifth is evaluated by every rank
    # after reduction, so it must be added once to each rank's global totals.
    logits = torch.tensor([[3., 0.], [0., 3.], [0., 3.], [3., 0.], [3., 0.]])
    targets = torch.tensor([0, 1, 0, 0, 1])
    dataset = TensorDataset(logits, targets)
    sampler = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False, drop_last=True)
    loader = DataLoader(dataset, sampler=sampler, batch_size=2)
    args = build_parser().parse_args(["--label-mode", "binary", "--selection-metric", "f1"])
    args.num_classes, args.distributed, args.world_size = 2, True, 2
    args.workers, args.batch_size = 0, 2
    totals = iter([torch.tensor([300., 4.]), torch.tensor([400., 4.]), torch.tensor([[2, 1], [0, 1]])])
    monkeypatch.setattr("skyball_gameplay.training.dist.all_reduce", lambda tensor, op: tensor.copy_(next(totals)))
    accuracy = validate(loader, torch.nn.Identity(), torch.nn.CrossEntropyLoss(), args, torch.device("cpu"))
    assert accuracy == pytest.approx(60.)
    assert args.last_validation_metrics["count"] == 5
    assert args.last_validation_metrics["confusion"] == [[2, 1], [1, 1]]
    assert args.last_validation_metrics["f1"] == .5
