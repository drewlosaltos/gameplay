"""Self-contained training CLI for SkyBall gameplay classifiers."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
import warnings
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.datasets
import torchvision.models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import Subset

from .data import build_datasets
from .modeling import create_gameplay_model
from .modeling import normalization_for_backbone


BEST_ACC1: float | torch.Tensor = 0.0
BEST_SELECTION_SCORE = -1.0


def build_parser() -> argparse.ArgumentParser:
    """Build the legacy-compatible gameplay training command line."""
    model_names = sorted(
        name
        for name, value in torchvision.models.__dict__.items()
        if name.islower() and not name.startswith("__") and callable(value)
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", default="imagenet")
    parser.add_argument("-a", "--arch", default="resnet18", choices=model_names + ['qwen_vision'])
    parser.add_argument("--timm-model", default="")
    parser.add_argument('--qwen-snapshot', default='')
    parser.add_argument('--freeze-encoder', action='store_true')
    parser.add_argument("-j", "--workers", default=4, type=int)
    parser.add_argument("--epochs", default=90, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("-b", "--batch-size", default=256, type=int)
    parser.add_argument("--grad-accum-steps", default=1, type=int)
    parser.add_argument("--lr", "--learning-rate", default=1e-4, type=float, dest="lr")
    parser.add_argument("--backbone-lr-multiplier", default=1.0, type=float)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
    parser.add_argument("--lr-scheduler", default="step", choices=["step", "cosine"])
    parser.add_argument("--lr-step-size", default=30, type=int)
    parser.add_argument("--lr-gamma", default=0.1, type=float)
    parser.add_argument("--min-lr", default=0.0, type=float)
    parser.add_argument('--warmup-epochs', default=0.0, type=float)
    parser.add_argument('--max-grad-norm', default=0.0, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, dest="weight_decay")
    parser.add_argument("-p", "--print-freq", default=10, type=int)
    checkpoint_source = parser.add_mutually_exclusive_group()
    checkpoint_source.add_argument("--resume", default="")
    checkpoint_source.add_argument(
        "--init-checkpoint",
        default="",
        help="load model weights only and begin a fresh optimizer/schedule",
    )
    parser.add_argument("-e", "--evaluate", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--world-size", default=-1, type=int)
    parser.add_argument("--rank", default=-1, type=int)
    parser.add_argument("--dist-url", default="tcp://224.66.41.62:23456")
    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument("--no-accel", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument('--amp-dtype', choices=['float16', 'bfloat16'], default='float16')
    parser.add_argument("--multiprocessing-distributed", action="store_true")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--input-frames", default=1, type=int, choices=[1, 4])
    parser.add_argument("--label-mode", default="multiclass", choices=["binary", "multiclass"])
    parser.add_argument("--image-size", default=[224, 224], type=int, nargs=2)
    parser.add_argument("--crop-scale-min", default=0.08, type=float)
    parser.add_argument("--train-geometry", default="random_crop", choices=["random_crop", "full_frame"])
    parser.add_argument('--resize-mode', default='stretch', choices=['stretch', 'letterbox'])
    parser.add_argument('--resize-geometry', dest='resize_mode', choices=['stretch', 'letterbox'])
    parser.add_argument('--rotation-degrees', default=0.0, type=float)
    parser.add_argument("--selection-metric", default="accuracy", choices=["accuracy", "f1", "balanced_accuracy"])
    parser.add_argument("--gameplay-loss-weight", default=1.0, type=float)
    parser.add_argument("--validation-fp32", action="store_true")
    parser.add_argument("--color-jitter", default=0.0, type=float)
    parser.add_argument(
        "--aug-policy",
        default="none",
        choices=["none", "autoaugment", "randaugment", "trivialaugment", "augmix"],
    )
    parser.add_argument(
        "--auto-augment-policy",
        default="imagenet",
        choices=["imagenet", "cifar10", "svhn"],
    )
    parser.add_argument("--rand-augment-ops", default=2, type=int)
    parser.add_argument("--rand-augment-magnitude", default=9, type=int)
    parser.add_argument("--augmix-severity", default=3, type=int)
    parser.add_argument("--random-erasing", default=0.0, type=float)
    parser.add_argument("--temporal-dropout", default=0.0, type=float)
    parser.add_argument("--temporal-shared-augment", action="store_true")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument(
        "--max-train-batches",
        default=None,
        type=int,
        help="Optional bounded smoke-test limit; omit for normal training.",
    )
    parser.add_argument(
        "--max-val-batches",
        default=None,
        type=int,
        help="Optional bounded smoke-test limit; omit for normal evaluation.",
    )
    return parser


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    backbone_lr_multiplier: float = 1.0,
    optimizer_name: str = "adam",
) -> torch.optim.Optimizer:
    """Build Adam/AdamW with an optional conservative encoder rate."""
    if backbone_lr_multiplier <= 0:
        raise ValueError("backbone learning-rate multiplier must be positive")
    unwrapped = model.module if hasattr(model, "module") else model
    if backbone_lr_multiplier == 1.0 or not hasattr(unwrapped, "encoder"):
        parameter_groups: Any = model.parameters()
    else:
        backbone_parameters = list(unwrapped.encoder.parameters())
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        head_parameters = [
            parameter
            for parameter in unwrapped.parameters()
            if id(parameter) not in backbone_ids
        ]
        parameter_groups = [
            {
                "params": backbone_parameters,
                "lr": learning_rate * backbone_lr_multiplier,
            },
            {"params": head_parameters, "lr": learning_rate},
        ]
    optimizer_class = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    return optimizer_class(
        parameter_groups,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def should_optimizer_step(
    batch_index: int,
    batch_count: int,
    grad_accum_steps: int,
) -> bool:
    """Return whether an accumulation group ends at this microbatch."""
    if grad_accum_steps <= 0:
        raise ValueError("gradient accumulation steps must be positive")
    return (batch_index + 1) % grad_accum_steps == 0 or batch_index + 1 == batch_count


def gradient_accumulation_divisor(
    batch_index: int,
    batch_count: int,
    grad_accum_steps: int,
) -> int:
    """Scale the final partial accumulation group correctly."""
    if grad_accum_steps <= 0:
        raise ValueError("gradient accumulation steps must be positive")
    group_start = (batch_index // grad_accum_steps) * grad_accum_steps
    return min(grad_accum_steps, batch_count - group_start)


def main(argv: list[str] | None = None) -> None:
    """Launch local single-node or multiprocessing-distributed training."""
    args = build_parser().parse_args(argv)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn("Deterministic training may be slower and alter restart behavior.")
    if args.gpu is not None:
        warnings.warn("A specific GPU disables automatic data parallelism.")
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    device = current_device(args.no_accel)
    ngpus_per_node = torch.cuda.device_count() if device.type == "cuda" else 1
    print(f"Using device: {device}")
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def current_device(no_accel: bool) -> torch.device:
    """Resolve CUDA through the accelerator API with a CPU fallback."""
    if not no_accel and torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device("cpu")


def main_worker(gpu: int | None, ngpus_per_node: int, args: argparse.Namespace) -> None:
    """Run one training process."""
    global BEST_ACC1, BEST_SELECTION_SCORE
    BEST_ACC1, BEST_SELECTION_SCORE = 0.0, -1.0
    args.gpu = gpu
    if args.seed is not None:
        random.seed(args.seed + (gpu or 0))
        torch.manual_seed(args.seed + (gpu or 0))
        cudnn.deterministic = True
        cudnn.benchmark = False
    if args.label_mode != "binary" and (
        args.selection_metric != "accuracy" or args.gameplay_loss_weight != 1.0
    ):
        raise ValueError("Gameplay F1 selection and loss weighting require binary labels")
    if args.gameplay_loss_weight <= 0:
        raise ValueError("gameplay loss weight must be positive")
    device = current_device(args.no_accel)
    if device.type == "cuda" and gpu is not None:
        torch.cuda.set_device(gpu)
        device = torch.device("cuda", gpu)
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + int(gpu)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    args.normalization_mean, args.normalization_std = normalization_for_backbone(
        args.timm_model
    )
    if args.qwen_snapshot:
        if args.arch != 'qwen_vision':
            raise ValueError('--qwen-snapshot requires --arch qwen_vision')
        args.normalization_mean = args.normalization_std = (.5, .5, .5)
    if args.dummy:
        image_shape = (
            (args.input_frames, 3, *args.image_size)
            if args.input_frames == 4
            else (3, *args.image_size)
        )
        train_dataset = torchvision.datasets.FakeData(
            64,
            image_shape,
            2,
            transforms.ToTensor(),
        )
        val_dataset = torchvision.datasets.FakeData(
            16,
            image_shape,
            2,
            transforms.ToTensor(),
        )
        args.num_classes = 2
        args.classes = ["non_gameplay", "gameplay"]
    else:
        train_dataset, val_dataset = build_datasets(args)
        args.num_classes = len(train_dataset.classes)
        args.classes = list(train_dataset.classes)

    model = create_gameplay_model(
        arch=args.arch,
        timm_model=args.timm_model,
        input_frames=args.input_frames,
        num_classes=args.num_classes,
        temporal_dropout=args.temporal_dropout,
        pretrained=args.pretrained,
        qwen_snapshot=args.qwen_snapshot,
        image_size=args.image_size,
    )
    if args.freeze_encoder:
        if not hasattr(model, 'encoder'):
            raise ValueError('--freeze-encoder requires an encoder module')
        model.encoder.requires_grad_(False)
    if args.distributed:
        if device.type != "cuda":
            model.to(device)
            model = torch.nn.parallel.DistributedDataParallel(model)
        elif gpu is not None:
            model.to(device)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif device.type == "cuda":
        model = torch.nn.DataParallel(model).cuda()
    else:
        print("Using CPU; training will be slow")
        model.to(device)

    initialize_checkpoint(args, model, device)
    class_weight = (
        torch.tensor([1.0, args.gameplay_loss_weight], device=device)
        if args.gameplay_loss_weight != 1.0 else None
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)
    optimizer = build_optimizer(
        model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        backbone_lr_multiplier=args.backbone_lr_multiplier,
        optimizer_name=args.optimizer,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda" and args.amp_dtype == 'float16')
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        if args.lr_scheduler == "cosine"
        else StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    )
    resume_checkpoint(args, model, optimizer, scheduler, scaler, device)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            shuffle=False,
            drop_last=True,
        )
    else:
        train_sampler = None
        val_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        sampler=val_sampler,
    )
    if args.evaluate:
        validate(val_loader, model, criterion, args, device)
        if args.distributed:
            dist.destroy_process_group()
        return

    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_one_epoch(train_loader, model, criterion, optimizer, scaler, epoch, device, args)
        acc1 = validate(val_loader, model, criterion, args, device)
        scheduler.step()
        selection_score = args.last_validation_metrics[args.selection_metric]
        is_best = selection_score > BEST_SELECTION_SCORE
        BEST_SELECTION_SCORE = max(selection_score, BEST_SELECTION_SCORE)
        BEST_ACC1 = max(float(acc1), float(BEST_ACC1))
        if not args.multiprocessing_distributed or args.rank % ngpus_per_node == 0:
            tag = args.run_tag or f"{Path(args.data).name}_{args.arch}_f{args.input_frames}"
            save_checkpoint(
                checkpoint_payload(args, model, optimizer, scheduler, scaler, epoch),
                is_best,
                tag=tag,
                output_dir=args.output_dir,
            )
            with (Path(args.output_dir) / "epoch_metrics.jsonl").open("a") as handle:
                handle.write(json.dumps({"epoch": epoch + 1, "selected": is_best,
                    "selection_metric": args.selection_metric,
                    **args.last_validation_metrics}) + "\n")
    if args.distributed:
        dist.destroy_process_group()


def resume_checkpoint(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> None:
    """Restore complete training state when requested."""
    global BEST_ACC1, BEST_SELECTION_SCORE
    if not args.resume:
        return
    if not Path(args.resume).is_file():
        print(f"=> no checkpoint found at '{args.resume}'")
        return
    location = device if args.gpu is not None else "cpu"
    checkpoint = torch.load(
        args.resume,
        map_location=location,
        weights_only=False,
    )
    if checkpoint.get("selection_metric", "accuracy") != args.selection_metric:
        raise ValueError("Resume requires the checkpoint's original selection metric")
    args.start_epoch = int(checkpoint["epoch"])
    BEST_ACC1 = checkpoint["best_acc1"]
    BEST_SELECTION_SCORE = checkpoint.get("best_selection_score", float(BEST_ACC1) / 100.0)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")


def initialize_checkpoint(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    """Initialize model weights without restoring optimizer or epoch state."""
    if not args.init_checkpoint:
        return
    checkpoint_path = Path(args.init_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Initialization checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "timm_model": args.timm_model,
        "input_frames": args.input_frames,
        "label_mode": args.label_mode,
        "num_classes": args.num_classes,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Initialization checkpoint architecture mismatch: {mismatches}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    print(
        f"=> initialized model weights from '{checkpoint_path}' "
        f"(source epoch {checkpoint.get('epoch', 'unknown')})"
    )


def checkpoint_payload(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
) -> dict[str, Any]:
    """Build the historical checkpoint schema consumed by runtime tools."""
    return {
        "epoch": epoch + 1,
        "arch": args.arch,
        "timm_model": args.timm_model,
        "input_frames": args.input_frames,
        "label_mode": args.label_mode,
        "image_size": args.image_size,
        "crop_scale_min": args.crop_scale_min,
        "train_geometry": args.train_geometry,
        "resize_mode": getattr(args, 'resize_mode', 'stretch'),
        "resize_geometry": getattr(args, 'resize_mode', 'stretch'),
        "rotation_degrees": getattr(args, 'rotation_degrees', 0.0),
        "selection_metric": args.selection_metric,
        "best_selection_score": BEST_SELECTION_SCORE,
        "validation_metrics": getattr(args, "last_validation_metrics", None),
        "gameplay_loss_weight": args.gameplay_loss_weight,
        "validation_fp32": args.validation_fp32,
        "seed": args.seed,
        "color_jitter": args.color_jitter,
        "aug_policy": args.aug_policy,
        "auto_augment_policy": args.auto_augment_policy,
        "rand_augment_ops": args.rand_augment_ops,
        "rand_augment_magnitude": args.rand_augment_magnitude,
        "augmix_severity": args.augmix_severity,
        "random_erasing": args.random_erasing,
        "temporal_dropout": args.temporal_dropout,
        "temporal_shared_augment": args.temporal_shared_augment,
        "normalization_mean": args.normalization_mean,
        "normalization_std": args.normalization_std,
        "optimizer_name": args.optimizer,
        "learning_rate": args.lr,
        "backbone_lr_multiplier": args.backbone_lr_multiplier,
        "grad_accum_steps": args.grad_accum_steps,
        "warmup_epochs": getattr(args, 'warmup_epochs', 0.0),
        "max_grad_norm": getattr(args, 'max_grad_norm', 0.0),
        "init_checkpoint": args.init_checkpoint or None,
        "weight_decay": args.weight_decay,
        "amp": args.amp,
        "amp_dtype": getattr(args, 'amp_dtype', 'float16'),
        "qwen_snapshot": getattr(args, 'qwen_snapshot', ''),
        "freeze_encoder": getattr(args, 'freeze_encoder', False),
        "num_classes": args.num_classes,
        "classes": args.classes,
        "state_dict": model.state_dict(),
        "best_acc1": BEST_ACC1,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
    }


def train_one_epoch(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Train one epoch with exact partial-group accumulation scaling."""
    batch_count = effective_batch_count(loader, args.max_train_batches)
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    topk = AverageMeter(f"Acc@{min(5, args.num_classes)}", ":6.2f")
    progress = ProgressMeter(batch_count, [batch_time, data_time, losses, top1, topk], f"Epoch: [{epoch}]")
    model.train()
    if getattr(args, 'freeze_encoder', False):
        (model.module if hasattr(model, 'module') else model).encoder.eval()
    optimizer.zero_grad()
    epoch_learning_rates = [group['lr'] for group in optimizer.param_groups]
    end = time.time()
    for index, (images, target) in enumerate(loader):
        if index >= batch_count:
            break
        data_time.update(time.time() - end)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        warmup_epochs = getattr(args, 'warmup_epochs', 0.0)
        if warmup_epochs > 0:
            warmup_scale = min(1.0, (epoch + (index+1)/batch_count) / warmup_epochs)
            for group, learning_rate in zip(optimizer.param_groups, epoch_learning_rates):
                group['lr'] = learning_rate * warmup_scale
        from contextlib import nullcontext
        update_now = should_optimizer_step(index, batch_count, args.grad_accum_steps)
        synchronization = (model.no_sync() if hasattr(model, 'no_sync') and not update_now
                           else nullcontext())
        with synchronization:
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda',
                                dtype=getattr(torch, getattr(args, 'amp_dtype', 'float16'))):
                output = model(images)
                loss = criterion(output, target)
                backward_loss = loss / gradient_accumulation_divisor(
                    index,
                    batch_count,
                    args.grad_accum_steps,
                )
            scaler.scale(backward_loss).backward()
        max_k = min(5, args.num_classes)
        acc1, acck = accuracy(output, target, topk=(1, max_k))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        topk.update(acck[0], images.size(0))
        if update_now:
            if getattr(args, 'max_grad_norm', 0.0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        batch_time.update(time.time() - end)
        end = time.time()
        if index % args.print_freq == 0:
            progress.display(index + 1)
    for group, learning_rate in zip(optimizer.param_groups, epoch_learning_rates):
        group['lr'] = learning_rate


def effective_batch_count(loader: Any, requested: int | None) -> int:
    """Resolve an optional bounded smoke-test batch count."""
    if requested is None:
        return len(loader)
    if requested <= 0:
        raise ValueError("max batch limits must be positive")
    return min(len(loader), requested)


def validate(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    """Evaluate accuracy and binary confusion, including the residual DDP tail."""
    batch_count = effective_batch_count(loader, args.max_val_batches)
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    max_k = min(5, args.num_classes)
    topk = AverageMeter(f"Acc@{max_k}", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batch_count, [batch_time, losses, top1, topk], "Test: ")
    model.eval()
    confusion = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64, device=device)

    def run(active_loader: Any, limit: int, base_progress: int = 0) -> None:
        end = time.time()
        with torch.no_grad():
            for local_index, (images, target) in enumerate(active_loader):
                if local_index >= limit:
                    break
                index = base_progress + local_index
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    enabled=args.amp and device.type == "cuda" and not getattr(args, "validation_fp32", False),
                    dtype=getattr(torch, getattr(args, 'amp_dtype', 'float16')),
                ):
                    output = model(images)
                    loss = criterion(output, target)
                acc1, acck = accuracy(output, target, topk=(1, max_k))
                predicted = output.argmax(dim=1)
                confusion.add_(torch.bincount(
                    target * args.num_classes + predicted,
                    minlength=args.num_classes ** 2,
                ).reshape(args.num_classes, args.num_classes))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                topk.update(acck[0], images.size(0))
                batch_time.update(time.time() - end)
                end = time.time()
                if index % args.print_freq == 0:
                    progress.display(index + 1)

    run(loader, batch_count)
    if args.distributed:
        top1.all_reduce(device)
        topk.all_reduce(device)
        dist.all_reduce(confusion, dist.ReduceOp.SUM)
    full_validation = args.max_val_batches is None
    if (
        full_validation
        and args.distributed
        and len(loader.sampler) * args.world_size < len(loader.dataset)
    ):
        auxiliary = Subset(
            loader.dataset,
            range(len(loader.sampler) * args.world_size, len(loader.dataset)),
        )
        auxiliary_loader = torch.utils.data.DataLoader(
            auxiliary,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        run(auxiliary_loader, len(auxiliary_loader), len(loader))
    progress.display_summary()
    args.last_validation_metrics = metrics_from_confusion(confusion)
    print("Validation metrics: " + json.dumps(args.last_validation_metrics))
    return float(top1.avg)


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
    """Compute population metrics after DDP reduction and one residual tail."""
    counts = confusion.detach().cpu()
    total = int(counts.sum())
    metrics = {"accuracy": float(counts.diag().sum()) / max(total, 1),
               "count": total, "confusion": counts.tolist()}
    if counts.shape == (2, 2):
        tn, fp, fn, tp = counts.flatten().tolist()
        metrics.update(
            f1=2 * tp / max(2 * tp + fp + fn, 1),
            balanced_accuracy=.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
            precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
        )
    return metrics


def save_checkpoint(
    state: dict[str, Any],
    is_best: bool,
    *,
    tag: str,
    output_dir: str | Path,
) -> None:
    """Write last and best checkpoints using existing filename conventions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / f"last_{tag}.pth.tar"
    best_path = output_dir / f"best_{tag}.pth.tar"
    torch.save(state, last_path)
    if is_best:
        shutil.copyfile(last_path, best_path)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1


class AverageMeter:
    """Track a weighted streaming average."""

    def __init__(
        self,
        name: str,
        fmt: str = ":f",
        summary_type: Summary = Summary.NONE,
    ) -> None:
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self) -> None:
        self.val: Any = 0
        self.avg: Any = 0
        self.sum: Any = 0
        self.count = 0

    def update(self, value: Any, count: int = 1) -> None:
        self.val = value
        self.sum += value * count
        self.count += count
        self.avg = self.sum / self.count

    def all_reduce(self, device: torch.device) -> None:
        total = torch.tensor(
            [float(self.sum), self.count],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(total, dist.ReduceOp.SUM)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        template = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return template.format(**self.__dict__)

    def summary(self) -> str:
        if self.summary_type is Summary.AVERAGE:
            return f"{self.name} {float(self.avg):.3f}"
        return ""


class ProgressMeter:
    """Format batch progress in the established training-log form."""

    def __init__(self, num_batches: int, meters: list[AverageMeter], prefix: str = "") -> None:
        digits = len(str(num_batches))
        self.batch_format = "[{:" + str(digits) + "d}/" + str(num_batches) + "]"
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        print("\t".join([self.prefix + self.batch_format.format(batch), *map(str, self.meters)]))

    def display_summary(self) -> None:
        print(" ".join([" *", *(meter.summary() for meter in self.meters)]))


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: tuple[int, ...] = (1,),
) -> list[torch.Tensor]:
    """Compute top-k classification accuracy percentages."""
    with torch.no_grad():
        max_k = max(topk)
        batch_size = target.size(0)
        prediction = output.topk(max_k, 1, True, True).indices.t()
        correct = prediction.eq(target.view(1, -1).expand_as(prediction))
        return [
            correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size)
            for k in topk
        ]


if __name__ == "__main__":
    main()
