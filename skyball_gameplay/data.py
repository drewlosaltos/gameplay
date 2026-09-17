"""Manifest-driven datasets and transforms for SkyBall gameplay training."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader


def load_classes(
    root: str | Path,
    label_mode: str,
) -> tuple[list[str], dict[str, int]]:
    """Load ordered target names for binary or original-label training."""
    if label_mode == "binary":
        return ["non_gameplay", "gameplay"], {"non_gameplay": 0, "gameplay": 1}
    class_to_idx = json.loads(
        (Path(root) / "metadata" / "class_to_idx.json").read_text(encoding="utf-8")
    )
    classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    return classes, class_to_idx


def resolve_metadata_path(root: str | Path, relative_path: str) -> Path:
    """Resolve modern train paths while retaining legacy ``trn`` support."""
    root = Path(root)
    path = root / relative_path
    if path.exists():
        return path
    if relative_path.startswith("trn/"):
        train_path = root / "train" / relative_path.removeprefix("trn/")
        if train_path.exists():
            return train_path
    return path


def record_matches_split(record: dict[str, Any], split: str) -> bool:
    """Return whether a manifest record belongs to the requested split."""
    record_split = record.get("split")
    if split == "trn":
        return record_split in {"trn", "train"}
    if split == "train":
        return record_split in {"trn", "train"}
    return record_split == split


def record_target(
    record: dict[str, Any],
    class_to_idx: dict[str, int],
    label_mode: str,
) -> int:
    """Resolve a binary or original-class target from one manifest row."""
    if label_mode == "binary":
        return int(record["binary_label"])
    return int(class_to_idx[record["class_name"]])


class GameplayFrameDataset(torch.utils.data.Dataset):
    """Lazy image-level dataset backed by ``metadata/images.jsonl``."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Any = None,
        label_mode: str = "multiclass",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.label_mode = label_mode
        self.classes, self.class_to_idx = load_classes(root, label_mode)
        self.samples: list[tuple[str, int]] = []
        images_path = self.root / "metadata" / "images.jsonl"
        with images_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record_matches_split(record, split):
                    self.samples.append(
                        (
                            record["path"],
                            record_target(record, self.class_to_idx, label_mode),
                        )
                    )
        if not self.samples:
            raise RuntimeError(f"No {split} frame samples found in {images_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, target = self.samples[index]
        image = default_loader(resolve_metadata_path(self.root, image_path))
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class TemporalWindowDataset(torch.utils.data.Dataset):
    """Lazy fixed-window dataset backed by ``metadata/windows.jsonl``."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Any = None,
        input_frames: int = 4,
        label_mode: str = "multiclass",
        temporal_shared_augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.input_frames = input_frames
        self.label_mode = label_mode
        self.temporal_shared_augment = temporal_shared_augment
        self.classes, self.class_to_idx = load_classes(root, label_mode)
        self.samples: list[tuple[list[str], int]] = []
        windows_path = self.root / "metadata" / "windows.jsonl"
        with windows_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if not record_matches_split(record, split):
                    continue
                frame_paths = list(record["frame_paths"][:input_frames])
                if len(frame_paths) != input_frames:
                    continue
                self.samples.append(
                    (
                        frame_paths,
                        record_target(record, self.class_to_idx, label_mode),
                    )
                )
        if not self.samples:
            raise RuntimeError(f"No {split} temporal samples found in {windows_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        frame_paths, target = self.samples[index]
        frames = []
        shared_torch_state = None
        shared_random_state = None
        if self.transform is not None and self.temporal_shared_augment:
            shared_torch_state = torch.random.get_rng_state()
            shared_random_state = random.getstate()
        for frame_path in frame_paths:
            image = default_loader(resolve_metadata_path(self.root, frame_path))
            if self.transform is not None:
                if self.temporal_shared_augment:
                    torch.random.set_rng_state(shared_torch_state)
                    random.setstate(shared_random_state)
                image = self.transform(image)
            frames.append(image)
        return torch.stack(frames, dim=0), target


class EvaluationDataset(torch.utils.data.Dataset):
    """Evaluation dataset that retains source manifest records for reporting."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        input_frames: int,
        label_mode: str,
        transform: Any,
    ) -> None:
        self.root = Path(root)
        self.input_frames = input_frames
        self.transform = transform
        self.classes, self.class_to_idx = load_classes(root, label_mode)
        self.records: list[dict[str, Any]] = []
        metadata_name = "windows.jsonl" if input_frames == 4 else "images.jsonl"
        metadata_path = self.root / "metadata" / metadata_name
        with metadata_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if not record_matches_split(record, split):
                    continue
                record["target"] = record_target(record, self.class_to_idx, label_mode)
                self.records.append(record)
        if not self.records:
            raise RuntimeError(f"No records found for {split} in {metadata_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record = self.records[index]
        if self.input_frames == 4:
            frames = [
                self.transform(default_loader(resolve_metadata_path(self.root, path)))
                for path in record["frame_paths"][: self.input_frames]
            ]
            image = torch.stack(frames, dim=0)
        else:
            image = self.transform(
                default_loader(resolve_metadata_path(self.root, record["path"]))
            )
        return image, int(record["target"]), index


def build_policy_transform(
    *,
    aug_policy: str,
    auto_augment_policy: str,
    rand_augment_ops: int,
    rand_augment_magnitude: int,
    augmix_severity: int,
) -> Any:
    """Construct the optional PIL-level augmentation policy."""
    if aug_policy == "none":
        return None
    if aug_policy == "autoaugment":
        policy_map = {
            "imagenet": transforms.AutoAugmentPolicy.IMAGENET,
            "cifar10": transforms.AutoAugmentPolicy.CIFAR10,
            "svhn": transforms.AutoAugmentPolicy.SVHN,
        }
        return transforms.AutoAugment(policy=policy_map[auto_augment_policy])
    if aug_policy == "randaugment":
        return transforms.RandAugment(
            num_ops=rand_augment_ops,
            magnitude=rand_augment_magnitude,
        )
    if aug_policy == "trivialaugment":
        return transforms.TrivialAugmentWide()
    if aug_policy == "augmix":
        return transforms.AugMix(severity=augmix_severity)
    raise ValueError(f"Unsupported augmentation policy: {aug_policy}")

def build_transforms(args: Any) -> tuple[Any, Any]:
    """Build checkpoint-compatible train and validation transforms."""
    normalize = transforms.Normalize(
        mean=args.normalization_mean,
        std=args.normalization_std,
    )
    geometry = (
        transforms.Resize(tuple(args.image_size))
        if getattr(args, "train_geometry", "random_crop") == "full_frame"
        else transforms.RandomResizedCrop(
            tuple(args.image_size),
            scale=(args.crop_scale_min, 1.0),
        )
    )
    train_steps: list[Any] = [
        geometry,
        transforms.RandomHorizontalFlip(),
    ]
    policy = build_policy_transform(
        aug_policy=args.aug_policy,
        auto_augment_policy=args.auto_augment_policy,
        rand_augment_ops=args.rand_augment_ops,
        rand_augment_magnitude=args.rand_augment_magnitude,
        augmix_severity=args.augmix_severity,
    )
    if policy is not None:
        train_steps.append(policy)
    if args.color_jitter > 0:
        train_steps.append(
            transforms.ColorJitter(
                brightness=args.color_jitter,
                contrast=args.color_jitter,
                saturation=args.color_jitter,
                hue=min(0.5, args.color_jitter * 0.25),
            )
        )
    train_steps.extend([transforms.ToTensor(), normalize])
    if args.random_erasing > 0:
        train_steps.append(transforms.RandomErasing(p=args.random_erasing))
    return (
        transforms.Compose(train_steps),
        transforms.Compose(
            [
                transforms.Resize(tuple(args.image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )


def build_datasets(args: Any) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Build manifest-driven training and validation datasets."""
    train_transform, val_transform = build_transforms(args)
    if args.input_frames == 4:
        train_dataset = TemporalWindowDataset(
            args.data,
            "trn",
            transform=train_transform,
            input_frames=args.input_frames,
            label_mode=args.label_mode,
            temporal_shared_augment=args.temporal_shared_augment,
        )
        val_dataset = TemporalWindowDataset(
            args.data,
            "val",
            transform=val_transform,
            input_frames=args.input_frames,
            label_mode=args.label_mode,
        )
    elif args.label_mode == "binary":
        train_dataset = GameplayFrameDataset(
            args.data,
            "trn",
            transform=train_transform,
            label_mode=args.label_mode,
        )
        val_dataset = GameplayFrameDataset(
            args.data,
            "val",
            transform=val_transform,
            label_mode=args.label_mode,
        )
    else:
        train_dataset = torchvision.datasets.ImageFolder(
            Path(args.data) / "train",
            train_transform,
        )
        val_dataset = torchvision.datasets.ImageFolder(
            Path(args.data) / "val",
            val_transform,
        )
    return train_dataset, val_dataset
