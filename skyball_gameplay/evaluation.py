"""Evaluate SkyBall gameplay checkpoints on manifest-defined splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as functional
import torchvision.transforms as transforms

from .data import EvaluationDataset
from .data import load_classes
from .modeling import load_checkpoint_model
from .modeling import normalization_for_backbone
from .preprocessing import LetterboxResize


DEFAULT_PREFETCH_BUDGET_GIB = 8.0


def build_parser() -> argparse.ArgumentParser:
    """Build the local checkpoint-evaluation command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("-b", "--batch-size", default=256, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)
    parser.add_argument(
        "--prefetch-budget-gib",
        default=DEFAULT_PREFETCH_BUDGET_GIB,
        type=float,
        help="Maximum estimated host RAM for prefetched image tensors.",
    )
    parser.add_argument("--split", default="val", choices=["trn", "train", "val"])
    parser.add_argument("--eval-unit", default="native", choices=["native", "window"])
    parser.add_argument("--image-size", default=None, type=int, nargs=2)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--no-accel", action="store_true")
    return parser


def safe_div(numerator: float, denominator: float) -> float:
    """Divide while returning zero for an empty denominator."""
    return float(numerator / denominator) if denominator else 0.0


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    """Compute binary ROC AUC with exact tie handling and no sklearn dependency."""
    ordered = sorted(zip(y_score, y_true, strict=True), key=lambda row: row[0])
    positive_count = sum(y_true)
    negative_count = len(y_true) - positive_count
    if not positive_count or not negative_count:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(target for _, target in ordered[index:end])
        index = end
    return (
        rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
) -> dict[str, Any]:
    """Compute the complete binary metric contract."""
    confusion = binary_confusion(y_true, y_pred)
    tp, tn, fp, fn = (
        confusion["tp"],
        confusion["tn"],
        confusion["fp"],
        confusion["fn"],
    )
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "accuracy": safe_div(tp + tn, len(y_true)),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "roc_auc": roc_auc(y_true, y_score),
        "confusion": confusion,
        "count": len(y_true),
        "positive_count": sum(y_true),
        "negative_count": len(y_true) - sum(y_true),
    }


def binary_confusion(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, int]:
    """Count binary confusion cells."""
    tp = tn = fp = fn = 0
    for target, prediction in zip(y_true, y_pred, strict=True):
        if target == 1 and prediction == 1:
            tp += 1
        elif target == 0 and prediction == 0:
            tn += 1
        elif target == 0 and prediction == 1:
            fp += 1
        else:
            fn += 1
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def binary_metrics_at_threshold(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, Any]:
    """Compute threshold-sensitive binary metrics without recalculating AUC."""
    confusion = binary_confusion(y_true, y_pred)
    tp, tn, fp, fn = (
        confusion["tp"],
        confusion["tn"],
        confusion["fp"],
        confusion["fn"],
    )
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "accuracy": safe_div(tp + tn, len(y_true)),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "confusion": confusion,
    }


def threshold_sweep(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> dict[str, Any]:
    """Select best-F1 and best-balanced-accuracy thresholds."""
    thresholds = sorted(set(float(score) for score in y_score))
    if not thresholds:
        return {}
    ordered = sorted(zip(y_score, y_true, strict=True), key=lambda row: row[0])
    positive_count = sum(y_true)
    negative_count = len(y_true) - positive_count
    tp = positive_count
    fp = negative_count
    fn = tn = 0
    moved = 0
    best_f1 = None
    best_balanced = None
    for threshold in [0.0, *thresholds, 1.0]:
        while moved < len(ordered) and ordered[moved][0] < threshold:
            target = ordered[moved][1]
            if target == 1:
                tp -= 1
                fn += 1
            else:
                fp -= 1
                tn += 1
            moved += 1
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        row = {
            "threshold": threshold,
            "accuracy": safe_div(tp + tn, len(y_true)),
            "balanced_accuracy": 0.5 * (recall + specificity),
            "precision": precision,
            "recall": recall,
            "f1": safe_div(2 * precision * recall, precision + recall),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }
        if best_f1 is None or (row["f1"], row["balanced_accuracy"]) > (
            best_f1["f1"],
            best_f1["balanced_accuracy"],
        ):
            best_f1 = row
        if best_balanced is None or (row["balanced_accuracy"], row["f1"]) > (
            best_balanced["balanced_accuracy"],
            best_balanced["f1"],
        ):
            best_balanced = row
    return {"best_f1": best_f1, "best_balanced_accuracy": best_balanced}


def multiclass_confusion(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
) -> dict[str, Any]:
    """Build a row-target, column-prediction confusion matrix."""
    matrix = [[0 for _ in classes] for _ in classes]
    for target, prediction in zip(y_true, y_pred, strict=True):
        matrix[target][prediction] += 1
    return {"classes": list(classes), "matrix": matrix}


def grouped_binary_metrics(
    records: Sequence[dict[str, Any]],
    y_pred: Sequence[int],
    y_score: Sequence[float],
    key: str,
) -> dict[str, Any]:
    """Report binary metrics by one manifest field."""
    groups: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"true": [], "pred": [], "score": []}
    )
    for record, prediction, score in zip(records, y_pred, y_score, strict=True):
        group = groups[str(record.get(key, "unknown"))]
        group["true"].append(int(record["binary_label"]))
        group["pred"].append(int(prediction))
        group["score"].append(float(score))
    return {
        name: binary_metrics(values["true"], values["pred"], values["score"])
        for name, values in sorted(groups.items())
    }


def confusion_by_original_class(
    records: Sequence[dict[str, Any]],
    y_pred: Sequence[int],
) -> dict[str, Any]:
    """Break false gameplay admissions down by workbook class."""
    output: dict[str, dict[str, int]] = {}
    for record, prediction in zip(records, y_pred, strict=True):
        name = str(record.get("original_category") or record.get("class_name") or "unknown")
        stats = output.setdefault(
            name,
            {"count": 0, "predicted_gameplay": 0, "predicted_non_gameplay": 0},
        )
        stats["count"] += 1
        stats["predicted_gameplay" if prediction == 1 else "predicted_non_gameplay"] += 1
    return output


def binary_predictions_from_classes(
    classes: Sequence[str],
    predictions: Sequence[int],
) -> list[int]:
    """Project multiclass predictions onto active gameplay."""
    if "gameplay" not in classes:
        return [0] * len(predictions)
    gameplay_index = classes.index("gameplay")
    return [int(prediction == gameplay_index) for prediction in predictions]


def bounded_eval_workers(
    *,
    requested_workers: int,
    batch_size: int,
    input_frames: int,
    image_size: Sequence[int],
    prefetch_budget_gib: float,
) -> int:
    """Cap workers so one prefetched float tensor per worker fits the RAM budget."""
    if requested_workers <= 0:
        return 0
    if batch_size <= 0 or input_frames <= 0:
        raise ValueError("batch size and input frames must be positive")
    if len(image_size) != 2 or any(int(value) <= 0 for value in image_size):
        raise ValueError("image size must contain two positive dimensions")
    if prefetch_budget_gib <= 0:
        raise ValueError("prefetch budget must be positive")
    height, width = (int(value) for value in image_size)
    batch_bytes = batch_size * input_frames * 3 * height * width * 4
    budget_bytes = int(prefetch_budget_gib * 1024**3)
    budget_workers = max(1, budget_bytes // batch_bytes)
    return min(requested_workers, budget_workers)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run native/window evaluation and write metrics and optional predictions."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_accel = not args.no_accel and torch.accelerator.is_available()
    device = torch.accelerator.current_accelerator() if use_accel else torch.device("cpu")
    model, metadata = load_checkpoint_model(
        args.checkpoint, device, image_size_override=args.image_size,
    )
    raw_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_input_frames = int(metadata["input_frames"])
    label_mode = str(metadata["label_mode"])
    classes = list(raw_checkpoint.get("classes") or load_classes(args.data, label_mode)[0])
    num_classes = int(metadata["num_classes"])
    image_size = args.image_size or raw_checkpoint.get("image_size")
    mean = raw_checkpoint.get("normalization_mean")
    std = raw_checkpoint.get("normalization_std")
    del raw_checkpoint
    if mean is None or std is None:
        mean, std = normalization_for_backbone(str(metadata["timm_model"]))
    if image_size is None:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        image_size_for_metrics: Any = "legacy_resize256_centercrop224"
    else:
        transform = transforms.Compose(
            [
                LetterboxResize(image_size) if metadata['resize_mode']=='letterbox' else transforms.Resize(tuple(image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        image_size_for_metrics = list(image_size)
    eval_input_frames = 4 if args.eval_unit == "window" else model_input_frames
    dataset = EvaluationDataset(
        args.data,
        args.split,
        eval_input_frames,
        label_mode,
        transform,
    )
    eval_workers = bounded_eval_workers(
        requested_workers=args.workers,
        batch_size=args.batch_size,
        input_frames=eval_input_frames,
        image_size=image_size or [224, 224],
        prefetch_budget_gib=args.prefetch_budget_gib,
    )
    if eval_workers < args.workers:
        print(
            f"Capping evaluation workers from {args.workers} to {eval_workers} "
            f"for a {args.prefetch_budget_gib:g} GiB prefetch budget",
            flush=True,
        )
    loader_options: dict[str, Any] = {}
    if eval_workers > 0:
        loader_options["prefetch_factor"] = 1
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=eval_workers,
        pin_memory=use_accel,
        **loader_options,
    )
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    indices: list[int] = []
    with torch.inference_mode():
        for images, target, batch_indices in loader:
            images = images.to(device, non_blocking=True)
            if model_input_frames == 1 and images.ndim == 5:
                batch_size, num_frames, channels, height, width = images.shape
                logits = model(images.reshape(batch_size * num_frames, channels, height, width))
                probabilities = functional.softmax(logits, dim=1).reshape(
                    batch_size,
                    num_frames,
                    -1,
                ).mean(dim=1)
            else:
                probabilities = functional.softmax(model(images), dim=1)
            predictions = probabilities.argmax(dim=1)
            y_true.extend(int(value) for value in target.tolist())
            y_pred.extend(int(value) for value in predictions.cpu().tolist())
            if num_classes == 2:
                y_score.extend(float(value) for value in probabilities[:, 1].cpu().tolist())
            elif "gameplay" in classes:
                gameplay_index = classes.index("gameplay")
                y_score.extend(
                    float(value) for value in probabilities[:, gameplay_index].cpu().tolist()
                )
            else:
                y_score.extend(float("nan") for _ in range(images.size(0)))
            indices.extend(int(value) for value in batch_indices.tolist())
    records = [dataset.records[index] for index in indices]
    metrics: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "arch": metadata["arch"],
        "timm_model": metadata["timm_model"],
        "model_input_frames": model_input_frames,
        "eval_input_frames": eval_input_frames,
        "eval_unit": args.eval_unit,
        "label_mode": label_mode,
        "classes": classes,
        "image_size": image_size_for_metrics,
        "count": len(y_true),
        "accuracy": safe_div(sum(t == p for t, p in zip(y_true, y_pred, strict=True)), len(y_true)),
        "confusion_matrix": multiclass_confusion(y_true, y_pred, classes),
    }
    if label_mode == "binary":
        metrics.update(binary_report(records, y_true, y_pred, y_score))
    else:
        binary_true = [int(record["binary_label"]) for record in records]
        binary_pred = binary_predictions_from_classes(classes, y_pred)
        if not any(math.isnan(score) for score in y_score):
            metrics.update(binary_report(records, binary_true, binary_pred, y_score))
        metrics["binary_projection"] = {
            "positive_class": "gameplay",
            "negative_classes": [name for name in classes if name != "gameplay"],
        }
    write_json(args.output_dir / "metrics.json", metrics)
    if args.save_predictions:
        write_predictions(
            args.output_dir / "predictions.csv",
            indices,
            records,
            y_true,
            y_pred,
            y_score,
            classes,
            label_mode,
        )
    return metrics


def binary_report(
    records: Sequence[dict[str, Any]],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
) -> dict[str, Any]:
    """Assemble common binary and grouped report sections."""
    return {
        "binary": binary_metrics(y_true, y_pred, y_score),
        "threshold_sweep": threshold_sweep(y_true, y_score),
        "confusion_by_original_class": confusion_by_original_class(records, y_pred),
        "per_source_video": grouped_binary_metrics(records, y_pred, y_score, "source_video"),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic strict JSON."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_predictions(
    path: Path,
    indices: Sequence[int],
    records: Sequence[dict[str, Any]],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
    classes: Sequence[str],
    label_mode: str,
) -> None:
    """Write the established auditable per-record prediction CSV."""
    binary_pred = (
        list(y_pred)
        if label_mode == "binary"
        else binary_predictions_from_classes(classes, y_pred)
    )
    fieldnames = [
        "index",
        "target",
        "prediction",
        "score_gameplay",
        "target_class",
        "prediction_class",
        "binary_target",
        "binary_prediction",
        "class_name",
        "original_category",
        "source_video",
        "path",
        "window_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, record, target, prediction, projected, score in zip(
            indices,
            records,
            y_true,
            y_pred,
            binary_pred,
            y_score,
            strict=True,
        ):
            writer.writerow(
                {
                    "index": index,
                    "target": target,
                    "prediction": prediction,
                    "score_gameplay": "" if math.isnan(score) else score,
                    "target_class": classes[target] if target < len(classes) else "",
                    "prediction_class": classes[prediction] if prediction < len(classes) else "",
                    "binary_target": record.get("binary_label", ""),
                    "binary_prediction": projected,
                    "class_name": record.get("class_name", ""),
                    "original_category": record.get("original_category", ""),
                    "source_video": record.get("source_video", ""),
                    "path": record.get("path", ""),
                    "window_id": record.get("window_id", ""),
                }
            )


def main(argv: list[str] | None = None) -> None:
    """Run evaluation and print the complete report."""
    metrics = evaluate(build_parser().parse_args(argv))
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
