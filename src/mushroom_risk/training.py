from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import (
    MushroomRiskDataset,
    SAFE_RISKS,
    UNSAFE_RISKS,
    build_label_map,
    filter_samples,
    get_transforms,
    load_samples,
    split_samples,
)
from .models import MODEL_DISPLAY_NAMES, build_model, count_trainable_parameters
from .utils import ensure_dir, select_device, set_seed


@dataclass
class ExperimentConfig:
    dataset_id: str = "derekkunowilliams/mushrooms"
    model_names: tuple[str, ...] = (
        "resnet50",
        "resnext50_32x4d",
        "densenet121",
        "seresnet50",
    )
    image_size: int = 224
    batch_size: int = 8
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 4
    seed: int = 42
    num_workers: int = 0
    min_images_per_class: int = 3
    max_images_per_class: int | None = None
    output_dir: str = "outputs/mushroom_comparison"
    pretrained: bool = True
    optimizer_name: str = "adamw"
    label_smoothing: float = 0.05
    use_weighted_sampler: bool = True
    use_class_weights: bool = False


def create_dataloaders(
    splits: dict[str, list],
    label_to_index: dict[str, int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    use_weighted_sampler: bool,
) -> dict[str, DataLoader]:
    train_transform, eval_transform = get_transforms(image_size)
    datasets = {
        "train": MushroomRiskDataset(splits["train"], label_to_index, train_transform),
        "val": MushroomRiskDataset(splits["val"], label_to_index, eval_transform),
        "test": MushroomRiskDataset(splits["test"], label_to_index, eval_transform),
    }

    sampler = None
    if use_weighted_sampler:
        train_labels = [sample.risk for sample in splits["train"]]
        counts = Counter(train_labels)
        sample_weights = [1.0 / counts[label] for label in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train" and sampler is None),
            sampler=sampler if split == "train" else None,
            num_workers=num_workers,
        )
        for split, dataset in datasets.items()
    }


def compute_class_weights(
    train_samples: list,
    label_to_index: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    counts = Counter(sample.risk for sample in train_samples)
    total = sum(counts.values())
    num_classes = len(label_to_index)
    weights = torch.ones(num_classes, dtype=torch.float32)
    for label, index in label_to_index.items():
        weights[index] = total / (num_classes * counts[label])
    return weights.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def dangerous_error_rate(
    target_indices: list[int],
    predicted_indices: list[int],
    class_names: list[str],
) -> tuple[int, float]:
    target_labels = [class_names[index] for index in target_indices]
    predicted_labels = [class_names[index] for index in predicted_indices]

    unsafe_total = sum(label in UNSAFE_RISKS for label in target_labels)
    dangerous_errors = sum(
        true_label in UNSAFE_RISKS and predicted_label in SAFE_RISKS
        for true_label, predicted_label in zip(target_labels, predicted_labels)
    )
    rate = dangerous_errors / unsafe_total if unsafe_total else 0.0
    return dangerous_errors, rate


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        all_targets,
        all_predictions,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    dangerous_errors, dangerous_rate = dangerous_error_rate(
        all_targets,
        all_predictions,
        class_names,
    )

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "balanced_accuracy": balanced_accuracy_score(all_targets, all_predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "dangerous_errors": dangerous_errors,
        "dangerous_error_rate": dangerous_rate,
        "targets": all_targets,
        "predictions": all_predictions,
        "per_class_metrics": [
            {
                "class_name": class_names[index],
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(per_class_support[index]),
            }
            for index in range(len(class_names))
        ],
    }


def model_selection_score(metrics: dict[str, Any]) -> float:
    return (0.6 * float(metrics["f1_macro"])) + (0.4 * float(metrics["balanced_accuracy"]))


def plot_history(history: dict[str, list[float]], output_path: Path, title: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Val loss")
    axes[0].set_title(f"{title} loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train accuracy")
    axes[1].plot(epochs, history["val_accuracy"], label="Val accuracy")
    axes[1].plot(epochs, history["val_f1"], label="Val macro F1")
    axes[1].plot(epochs, history["val_balanced_accuracy"], label="Val balanced acc")
    axes[1].set_title(f"{title} metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(
    targets: list[int],
    predictions: list[int],
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_per_class_metrics(
    model_name: str,
    per_class_metrics: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_path = output_dir / f"{model_name}_per_class_metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_name", "precision", "recall", "f1", "support"],
        )
        writer.writeheader()
        writer.writerows(per_class_metrics)


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    if not results:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def plot_model_comparison(results: list[dict[str, Any]], output_path: Path) -> None:
    if not results:
        return

    labels = [result["display_name"] for result in results]
    metrics = [
        ("test_accuracy", "Test accuracy", "higher is better"),
        ("test_balanced_accuracy", "Balanced accuracy", "higher is better"),
        ("test_f1_macro", "Macro F1", "higher is better"),
        ("dangerous_error_rate", "Dangerous error rate", "lower is better"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    colors = ["#31572c", "#4f772d", "#90a955", "#ecf39e"]
    for ax, (metric_key, title, subtitle) in zip(axes.ravel(), metrics):
        values = [float(result[metric_key]) for result in results]
        bars = ax.bar(labels, values, color=colors[: len(values)])
        ax.set_title(f"{title} ({subtitle})")
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.025, 0.98),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Mushroom risk model comparison", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def train_model(
    model_name: str,
    dataloaders: dict[str, DataLoader],
    class_names: list[str],
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
    class_weights: torch.Tensor | None,
) -> dict[str, Any]:
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    print(f"\nStarting {display_name} on {device.type}", flush=True)

    model = build_model(model_name, len(class_names), pretrained=config.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.label_smoothing,
    )

    if config.optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError("Only adamw is supported in the clean rebuild.")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "val_f1": [],
        "val_balanced_accuracy": [],
    }
    best_score = -1.0
    best_state = None
    stale_epochs = 0
    checkpoint_path = output_dir / f"{model_name}_best.pt"

    for epoch in range(config.epochs):
        start = time.time()
        train_loss, train_accuracy = train_one_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            device,
        )
        val_metrics = evaluate(model, dataloaders["val"], criterion, device, class_names)
        elapsed = time.time() - start
        score = model_selection_score(val_metrics)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_macro"])
        history["val_balanced_accuracy"].append(val_metrics["balanced_accuracy"])

        print(
            f"[{display_name}] Epoch {epoch + 1}/{config.epochs} | "
            f"train_acc={train_accuracy:.3f} "
            f"val_acc={val_metrics['accuracy']:.3f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.3f} "
            f"val_f1={val_metrics['f1_macro']:.3f} "
            f"dangerous={val_metrics['dangerous_error_rate']:.3f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_name": model_name,
                    "class_names": class_names,
                    "image_size": config.image_size,
                    "state_dict": best_state,
                },
                checkpoint_path,
            )
            stale_epochs = 0
        else:
            stale_epochs += 1

        scheduler.step()

        if stale_epochs >= config.patience:
            print(f"[{display_name}] Early stopping.", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint was saved for {model_name}")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, dataloaders["test"], criterion, device, class_names)

    plot_history(history, output_dir / f"{model_name}_history.png", display_name)
    plot_confusion(
        test_metrics["targets"],
        test_metrics["predictions"],
        class_names,
        output_dir / f"{model_name}_confusion.png",
        f"{display_name} confusion matrix",
    )
    save_per_class_metrics(model_name, test_metrics["per_class_metrics"], output_dir)

    print(
        f"[{display_name}] Test acc={test_metrics['accuracy']:.3f} "
        f"balanced_acc={test_metrics['balanced_accuracy']:.3f} "
        f"macro_f1={test_metrics['f1_macro']:.3f} "
        f"dangerous_error_rate={test_metrics['dangerous_error_rate']:.3f}",
        flush=True,
    )

    return {
        "model_name": model_name,
        "display_name": display_name,
        "parameters": count_trainable_parameters(model),
        "best_val_score": best_score,
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "dangerous_errors": test_metrics["dangerous_errors"],
        "dangerous_error_rate": test_metrics["dangerous_error_rate"],
        "checkpoint_path": str(checkpoint_path),
    }


def run_experiment(config: ExperimentConfig, data_dir: str | None = None) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    dataset_root, raw_samples = load_samples(config.dataset_id, data_dir)
    samples = filter_samples(
        raw_samples,
        config.min_images_per_class,
        config.max_images_per_class,
        config.seed,
    )
    splits = split_samples(samples, config.seed)
    class_names, label_to_index = build_label_map(samples)
    dataloaders = create_dataloaders(
        splits,
        label_to_index,
        config.image_size,
        config.batch_size,
        config.num_workers,
        config.use_weighted_sampler,
    )

    device = select_device()
    class_weights = (
        compute_class_weights(splits["train"], label_to_index, device)
        if config.use_class_weights
        else None
    )

    metadata = {
        "config": asdict(config),
        "dataset_root": str(dataset_root),
        "num_samples": len(samples),
        "num_classes": len(class_names),
        "class_names": class_names,
        "split_sizes": {split: len(split_samples_) for split, split_samples_ in splits.items()},
        "class_distribution": Counter(sample.risk for sample in samples),
        "use_weighted_sampler": config.use_weighted_sampler,
        "use_class_weights": config.use_class_weights,
        "class_weights": (
            {
                label: float(class_weights[index].detach().cpu())
                for label, index in label_to_index.items()
            }
            if class_weights is not None
            else None
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(json.loads(json.dumps(metadata, default=str)), indent=2),
        encoding="utf-8",
    )

    print(f"Dataset root: {dataset_root}")
    print(f"Samples: {len(samples)}")
    print(f"Classes: {len(class_names)}")
    print(f"Split sizes: {metadata['split_sizes']}")
    print(f"Device: {device}")
    if class_weights is not None:
        print(f"Using class weights: {metadata['class_weights']}")

    results = [
        train_model(
            model_name,
            dataloaders,
            class_names,
            config,
            device,
            output_dir,
            class_weights,
        )
        for model_name in config.model_names
    ]

    save_results(results, output_dir / "results.csv")
    plot_model_comparison(results, output_dir / "model_comparison.png")

    return {
        "metadata": metadata,
        "results": results,
        "output_dir": str(output_dir),
    }
