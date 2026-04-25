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

def _wrap_loader(loader, desc=""):
    return loader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    SpeciesClassificationDataset,
    SpeciesSample,
    build_label_map,
    build_splits,
    filter_splits_to_species_subset,
    get_transforms,
    load_species_samples,
)
from .models import MODEL_DISPLAY_NAMES, build_model, count_trainable_parameters
from .risk import RISK_ORDER, SAFE_RISKS, UNSAFE_RISKS, load_risk_map, map_species_to_risk
from .utils import ensure_dir, select_device, set_seed


@dataclass
class ExperimentConfig:
    metadata_path: str | None = None
    train_metadata_path: str | None = None
    val_metadata_path: str | None = None
    test_metadata_path: str | None = None
    images_root: str | None = None
    path_column: str | None = None
    species_column: str | None = None
    split_column: str | None = None
    risk_map_path: str = "data/risk_map.csv"
    model_names: tuple[str, ...] = (
        "resnet50",
        "resnext50_32x4d",
        "seresnet50",
        "convnext_tiny",
    )
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 15
    min_epochs: int = 4
    stop_at_perfect_train: bool = True
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 42
    num_workers: int = 0
    top_species: int | None = 100
    min_images_per_species: int = 30
    max_images_per_species: int | None = None
    val_size: float = 0.15
    test_size: float = 0.15
    output_dir: str = "outputs/df20_species_project"
    pretrained: bool = True
    optimizer_name: str = "adamw"
    label_smoothing: float = 0.05
    use_weighted_sampler: bool = True
    use_class_weights: bool = False
    abstention_thresholds: tuple[float, ...] = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8)


def create_dataloaders(
    splits: dict[str, list[SpeciesSample]],
    label_to_index: dict[str, int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    use_weighted_sampler: bool,
) -> dict[str, DataLoader]:
    train_transform, eval_transform = get_transforms(image_size)
    datasets = {
        "train": SpeciesClassificationDataset(splits["train"], label_to_index, train_transform),
        "val": SpeciesClassificationDataset(splits["val"], label_to_index, eval_transform),
        "test": SpeciesClassificationDataset(splits["test"], label_to_index, eval_transform),
    }

    sampler = None
    if use_weighted_sampler:
        train_labels = [sample.species_key for sample in splits["train"]]
        counts = Counter(train_labels)
        sample_weights = [1.0 / counts[label] for label in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    return {
        split_name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train" and sampler is None),
            sampler=sampler if split_name == "train" else None,
            num_workers=num_workers,
        )
        for split_name, dataset in datasets.items()
    }


def compute_class_weights(
    train_samples: list[SpeciesSample],
    label_to_index: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    counts = Counter(sample.species_key for sample in train_samples)
    total = sum(counts.values())
    num_classes = len(label_to_index)
    weights = torch.ones(num_classes, dtype=torch.float32)
    for species_key, index in label_to_index.items():
        weights[index] = total / (num_classes * counts[species_key])
    return weights.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _indices in _wrap_loader(loader, desc="train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def _safe_balanced_accuracy(targets: list[int], predictions: list[int]) -> float:
    if not targets:
        return 0.0
    return float(balanced_accuracy_score(targets, predictions))


def _species_macro_f1(targets: list[str], predictions: list[str]) -> float:
    if not targets:
        return 0.0
    return float(
        precision_recall_fscore_support(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )[2]
    )


def compute_risk_metrics(prediction_records: list[dict[str, Any]]) -> dict[str, Any]:
    covered = [record for record in prediction_records if record["true_risk"] != "unknown"]
    if not covered:
        return {
            "risk_coverage": 0.0,
            "risk_accuracy": 0.0,
            "dangerous_errors": 0,
            "dangerous_error_rate": 0.0,
            "risk_labels": [],
            "risk_matrix": [],
        }

    risk_labels = [label for label in RISK_ORDER if label in {record["true_risk"] for record in covered} | {record["pred_risk"] for record in covered}]
    true_risks = [record["true_risk"] for record in covered]
    pred_risks = [record["pred_risk"] for record in covered]
    unsafe_total = sum(true_risk in UNSAFE_RISKS for true_risk in true_risks)
    dangerous_errors = sum(
        true_risk in UNSAFE_RISKS and pred_risk in SAFE_RISKS
        for true_risk, pred_risk in zip(true_risks, pred_risks)
    )

    return {
        "risk_coverage": len(covered) / len(prediction_records),
        "risk_accuracy": sum(true_risk == pred_risk for true_risk, pred_risk in zip(true_risks, pred_risks)) / len(covered),
        "dangerous_errors": dangerous_errors,
        "dangerous_error_rate": dangerous_errors / unsafe_total if unsafe_total else 0.0,
        "risk_labels": risk_labels,
        "risk_matrix": confusion_matrix(true_risks, pred_risks, labels=risk_labels).tolist(),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_keys: list[str],
    species_display_lookup: dict[str, str],
    risk_map: dict[str, str],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    top3_correct = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    prediction_records: list[dict[str, Any]] = []
    dataset = loader.dataset

    for images, labels, indices in _wrap_loader(loader, desc="eval"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predictions = probabilities.max(dim=1)
        topk = probabilities.topk(k=min(3, len(class_keys)), dim=1)

        total_loss += criterion(outputs, labels).item() * images.size(0)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        top3_correct += (
            (topk.indices == labels.unsqueeze(1)).any(dim=1).sum().item()
        )

        predicted_indices = predictions.cpu().tolist()
        target_indices = labels.cpu().tolist()
        confidence_values = confidences.cpu().tolist()
        topk_indices = topk.indices.cpu().tolist()
        dataset_indices = indices.tolist()

        all_targets.extend(target_indices)
        all_predictions.extend(predicted_indices)

        for dataset_index, target_index, prediction_index, confidence_value, topk_prediction in zip(
            dataset_indices,
            target_indices,
            predicted_indices,
            confidence_values,
            topk_indices,
        ):
            sample = dataset.samples[dataset_index]
            true_species_key = class_keys[target_index]
            pred_species_key = class_keys[prediction_index]
            prediction_records.append(
                {
                    "image_path": str(sample.path),
                    "observation_id": sample.observation_id or "",
                    "true_species_key": true_species_key,
                    "true_species_name": species_display_lookup.get(true_species_key, true_species_key),
                    "pred_species_key": pred_species_key,
                    "pred_species_name": species_display_lookup.get(pred_species_key, pred_species_key),
                    "confidence": float(confidence_value),
                    "true_risk": map_species_to_risk(true_species_key, risk_map),
                    "pred_risk": map_species_to_risk(pred_species_key, risk_map),
                    "top3_species_keys": "|".join(class_keys[index] for index in topk_prediction),
                    "top3_species_names": "|".join(species_display_lookup.get(class_keys[index], class_keys[index]) for index in topk_prediction),
                }
            )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        all_targets,
        all_predictions,
        labels=list(range(len(class_keys))),
        zero_division=0,
    )
    risk_metrics = compute_risk_metrics(prediction_records)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "top3_accuracy": top3_correct / total,
        "balanced_accuracy": _safe_balanced_accuracy(all_targets, all_predictions),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "targets": all_targets,
        "predictions": all_predictions,
        "prediction_records": prediction_records,
        "per_class_metrics": [
            {
                "class_key": class_keys[index],
                "class_name": species_display_lookup.get(class_keys[index], class_keys[index]),
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(per_class_support[index]),
            }
            for index in range(len(class_keys))
        ],
        **risk_metrics,
    }


def model_selection_score(metrics: dict[str, Any]) -> float:
    return (0.7 * float(metrics["f1_macro"])) + (0.3 * float(metrics["balanced_accuracy"]))


def compute_top_confusions(prediction_records: list[dict[str, Any]], limit: int = 25) -> list[dict[str, Any]]:
    confusion_counter = Counter(
        (record["true_species_name"], record["pred_species_name"])
        for record in prediction_records
        if record["true_species_key"] != record["pred_species_key"]
    )
    return [
        {
            "true_species_name": true_species,
            "pred_species_name": pred_species,
            "count": count,
        }
        for (true_species, pred_species), count in confusion_counter.most_common(limit)
    ]


def compute_abstention_table(
    prediction_records: list[dict[str, Any]],
    thresholds: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(prediction_records)
    for threshold in thresholds:
        retained = [record for record in prediction_records if record["confidence"] >= threshold]
        if not retained:
            rows.append(
                {
                    "threshold": threshold,
                    "coverage": 0.0,
                    "retained_accuracy": 0.0,
                    "retained_macro_f1": 0.0,
                    "retained_risk_accuracy": 0.0,
                    "retained_dangerous_error_rate": 0.0,
                }
            )
            continue

        target_species = [record["true_species_key"] for record in retained]
        predicted_species = [record["pred_species_key"] for record in retained]
        retained_accuracy = sum(
            true_species == pred_species
            for true_species, pred_species in zip(target_species, predicted_species)
        ) / len(retained)
        retained_macro_f1 = _species_macro_f1(target_species, predicted_species)
        retained_risk_metrics = compute_risk_metrics(retained)

        rows.append(
            {
                "threshold": threshold,
                "coverage": len(retained) / total,
                "retained_accuracy": retained_accuracy,
                "retained_macro_f1": retained_macro_f1,
                "retained_risk_accuracy": retained_risk_metrics["risk_accuracy"],
                "retained_dangerous_error_rate": retained_risk_metrics["dangerous_error_rate"],
            }
        )
    return rows


def plot_history(history: dict[str, list[float]], output_path: Path, title: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Val loss")
    axes[0].set_title(f"{title} loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train top-1")
    axes[1].plot(epochs, history["val_accuracy"], label="Val top-1")
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
    display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_risk_confusion(
    risk_labels: list[str],
    risk_matrix: list[list[int]],
    output_path: Path,
    title: str,
) -> None:
    if not risk_labels or not risk_matrix:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(np.asarray(risk_matrix), display_labels=risk_labels)
    display.plot(ax=ax, cmap="Oranges", colorbar=False, xticks_rotation=30)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_model_comparison(results: list[dict[str, Any]], output_path: Path) -> None:
    if not results:
        return

    labels = [result["display_name"] for result in results]
    metrics = [
        ("test_accuracy", "Test top-1 accuracy"),
        ("test_top3_accuracy", "Test top-3 accuracy"),
        ("test_f1_macro", "Macro F1"),
        ("dangerous_error_rate", "Dangerous error rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    colors = ["#386641", "#6a994e", "#a7c957", "#f2e8cf"]
    for axis, (metric_key, title) in zip(axes.ravel(), metrics):
        values = [float(result[metric_key]) for result in results]
        bars = axis.bar(labels, values, color=colors[: len(values)])
        axis.set_title(title)
        axis.set_ylim(0.0, 1.0)
        axis.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.02, 0.98),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Nordic mushroom species model comparison", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_abstention_comparison(
    all_abstention_rows: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    if not all_abstention_rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for model_name, rows in all_abstention_rows.items():
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        thresholds = [row["threshold"] for row in rows]
        coverages = [row["coverage"] for row in rows]
        dangerous_rates = [row["retained_dangerous_error_rate"] for row in rows]
        axes[0].plot(thresholds, coverages, marker="o", label=display_name)
        axes[1].plot(thresholds, dangerous_rates, marker="o", label=display_name)

    axes[0].set_title("Coverage vs confidence threshold")
    axes[0].set_xlabel("Confidence threshold")
    axes[0].set_ylabel("Coverage")
    axes[0].set_ylim(0.0, 1.02)

    axes[1].set_title("Dangerous error rate vs threshold")
    axes[1].set_xlabel("Confidence threshold")
    axes[1].set_ylabel("Dangerous error rate")
    axes[1].set_ylim(0.0, 1.02)

    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def split_distribution_rows(splits: dict[str, list[SpeciesSample]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_species = sorted({sample.species_key for split_samples in splits.values() for sample in split_samples})
    display_lookup = {}
    for split_samples in splits.values():
        for sample in split_samples:
            display_lookup.setdefault(sample.species_key, sample.species_name)

    for species_key in all_species:
        row = {
            "species_key": species_key,
            "species_name": display_lookup.get(species_key, species_key),
        }
        total = 0
        for split_name in ("train", "val", "test"):
            count = sum(sample.species_key == species_key for sample in splits[split_name])
            row[f"{split_name}_count"] = count
            total += count
        row["total_count"] = total
        rows.append(row)

    rows.sort(key=lambda row: (-row["total_count"], row["species_key"]))
    return rows


def split_summary_rows(splits: dict[str, list[SpeciesSample]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, split_samples in splits.items():
        counts = Counter(sample.species_key for sample in split_samples)
        observation_ids = {sample.observation_id for sample in split_samples if sample.observation_id}
        max_count = max(counts.values()) if counts else 0
        min_count = min(counts.values()) if counts else 0
        rows.append(
            {
                "split": split_name,
                "num_images": len(split_samples),
                "num_species": len(counts),
                "num_observations": len(observation_ids),
                "largest_species_count": max_count,
                "smallest_species_count": min_count,
                "imbalance_ratio": (max_count / min_count) if min_count else 0.0,
            }
        )
    return rows


def observation_overlap_counts(splits: dict[str, list[SpeciesSample]]) -> dict[str, int]:
    split_to_observations = {
        split_name: {sample.observation_id for sample in split_samples if sample.observation_id}
        for split_name, split_samples in splits.items()
    }
    return {
        "train_val": len(split_to_observations["train"] & split_to_observations["val"]),
        "train_test": len(split_to_observations["train"] & split_to_observations["test"]),
        "val_test": len(split_to_observations["val"] & split_to_observations["test"]),
    }


def train_model(
    model_name: str,
    dataloaders: dict[str, DataLoader],
    class_keys: list[str],
    species_display_lookup: dict[str, str],
    config: ExperimentConfig,
    device: torch.device,
    output_paths: dict[str, Path],
    class_weights: torch.Tensor | None,
    risk_map: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    params = count_trainable_parameters(
        build_model(model_name, len(class_keys), pretrained=False)
    ) / 1e6
    print(f"\n{'═' * 62}", flush=True)
    print(f"  {display_name}  ({params:.1f}M params)  —  device: {device.type}", flush=True)
    print(f"{'═' * 62}", flush=True)
    print(f"  {'Ep':>3}  {'Train':>6}  {'Val':>6}  {'Top3':>6}  {'F1':>6}  {'Danger':>6}  {'Time':>7}", flush=True)
    print(f"  {'─'*3}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}", flush=True)

    model = build_model(model_name, len(class_keys), pretrained=config.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.label_smoothing,
    )
    if config.optimizer_name != "adamw":
        raise ValueError("Only AdamW is supported in this project scaffold.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
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
    checkpoint_path = output_paths["checkpoints"] / f"{model_name}_best.pt"

    for epoch in range(config.epochs):
        start = time.time()
        train_loss, train_accuracy = train_one_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            device,
        )
        val_metrics = evaluate(
            model,
            dataloaders["val"],
            criterion,
            device,
            class_keys,
            species_display_lookup,
            risk_map,
        )
        elapsed = time.time() - start
        score = model_selection_score(val_metrics)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_macro"])
        history["val_balanced_accuracy"].append(val_metrics["balanced_accuracy"])

        mins, secs = divmod(int(elapsed), 60)
        improved = score > best_score
        print(
            f"  {epoch + 1:>3}  "
            f"{train_accuracy:>5.1%}  "
            f"{val_metrics['accuracy']:>5.1%}  "
            f"{val_metrics['top3_accuracy']:>5.1%}  "
            f"{val_metrics['f1_macro']:>5.1%}  "
            f"{val_metrics['dangerous_error_rate']:>5.1%}  "
            f"{mins:>4}m{secs:02d}s"
            f"{' ✓' if improved else ''}",
            flush=True,
        )

        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_name": model_name,
                    "class_keys": class_keys,
                    "species_display_lookup": species_display_lookup,
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

        reached_perfect_train = (
            config.stop_at_perfect_train
            and epoch + 1 >= config.min_epochs
            and train_accuracy >= 0.9999
        )
        if reached_perfect_train:
            print(
                f"[{display_name}] Training accuracy reached 100%; stopping this model.",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint was saved for {model_name}")

    model.load_state_dict(best_state)
    test_metrics = evaluate(
        model,
        dataloaders["test"],
        criterion,
        device,
        class_keys,
        species_display_lookup,
        risk_map,
    )

    plot_history(history, output_paths["figures"] / f"{model_name}_history.png", display_name)
    plot_risk_confusion(
        test_metrics["risk_labels"],
        test_metrics["risk_matrix"],
        output_paths["figures"] / f"{model_name}_risk_confusion.png",
        f"{display_name} risk confusion",
    )

    save_csv(test_metrics["per_class_metrics"], output_paths["tables"] / f"{model_name}_per_class_metrics.csv")
    save_csv(test_metrics["prediction_records"], output_paths["predictions"] / f"{model_name}_predictions.csv")
    save_csv(
        compute_top_confusions(test_metrics["prediction_records"]),
        output_paths["tables"] / f"{model_name}_top_confusions.csv",
    )
    abstention_rows = compute_abstention_table(
        test_metrics["prediction_records"],
        config.abstention_thresholds,
    )
    save_csv(abstention_rows, output_paths["tables"] / f"{model_name}_abstention.csv")

    print(f"\n  TEST RESULTS — {display_name}", flush=True)
    print(f"  {'Top-1':>10}  {'Top-3':>10}  {'Macro F1':>10}  {'Danger':>10}", flush=True)
    print(
        f"  {test_metrics['accuracy']:>10.1%}  "
        f"{test_metrics['top3_accuracy']:>10.1%}  "
        f"{test_metrics['f1_macro']:>10.1%}  "
        f"{test_metrics['dangerous_error_rate']:>10.1%}",
        flush=True,
    )

    return (
        {
            "model_name": model_name,
            "display_name": display_name,
            "parameters": count_trainable_parameters(model),
            "best_val_score": best_score,
            "test_accuracy": test_metrics["accuracy"],
            "test_top3_accuracy": test_metrics["top3_accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_precision_macro": test_metrics["precision_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
            "test_f1_macro": test_metrics["f1_macro"],
            "risk_coverage": test_metrics["risk_coverage"],
            "risk_accuracy": test_metrics["risk_accuracy"],
            "dangerous_errors": test_metrics["dangerous_errors"],
            "dangerous_error_rate": test_metrics["dangerous_error_rate"],
            "checkpoint_path": str(checkpoint_path),
        },
        abstention_rows,
    )


def _serialize_config(config: ExperimentConfig) -> dict[str, Any]:
    payload = asdict(config)
    return payload


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    output_paths = {
        "figures": ensure_dir(output_dir / "figures"),
        "tables": ensure_dir(output_dir / "tables"),
        "predictions": ensure_dir(output_dir / "predictions"),
        "checkpoints": ensure_dir(output_dir / "checkpoints"),
    }

    raw_samples, metadata_stats = load_species_samples(
        metadata_path=config.metadata_path,
        train_metadata_path=config.train_metadata_path,
        val_metadata_path=config.val_metadata_path,
        test_metadata_path=config.test_metadata_path,
        images_root=config.images_root,
        path_column=config.path_column,
        species_column=config.species_column,
        split_column=config.split_column,
    )
    raw_splits, split_strategy = build_splits(
        raw_samples,
        seed=config.seed,
        val_size=config.val_size,
        test_size=config.test_size,
    )
    splits, selected_species_counts = filter_splits_to_species_subset(
        raw_splits,
        top_species=config.top_species,
        min_images_per_species=config.min_images_per_species,
        max_images_per_species=config.max_images_per_species,
        seed=config.seed,
    )
    class_keys, species_display_lookup, label_to_index = build_label_map(splits["train"])
    risk_map = load_risk_map(config.risk_map_path)
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
        "config": _serialize_config(config),
        "split_strategy": split_strategy,
        "metadata_stats": metadata_stats,
        "raw_num_samples": len(raw_samples),
        "selected_num_samples": sum(len(split_samples) for split_samples in splits.values()),
        "num_classes": len(class_keys),
        "class_keys": class_keys,
        "species_display_lookup": species_display_lookup,
        "split_sizes": {split_name: len(split_samples) for split_name, split_samples in splits.items()},
        "observation_overlap_counts": observation_overlap_counts(splits),
        "selected_species_counts_train": selected_species_counts,
        "use_weighted_sampler": config.use_weighted_sampler,
        "use_class_weights": config.use_class_weights,
        "class_weights": (
            {
                class_keys[index]: float(class_weights[index].detach().cpu())
                for index in range(len(class_keys))
            }
            if class_weights is not None
            else None
        ),
        "device": device.type,
        "risk_map_size": len(risk_map),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    save_csv(split_summary_rows(splits), output_paths["tables"] / "split_summary.csv")
    save_csv(split_distribution_rows(splits), output_paths["tables"] / "species_distribution.csv")

    results: list[dict[str, Any]] = []
    abstention_tables: dict[str, list[dict[str, Any]]] = {}
    for model_name in config.model_names:
        model_result, abstention_rows = train_model(
            model_name,
            dataloaders,
            class_keys,
            species_display_lookup,
            config,
            device,
            output_paths,
            class_weights,
            risk_map,
        )
        results.append(model_result)
        abstention_tables[model_name] = abstention_rows

    save_csv(results, output_paths["tables"] / "results.csv")
    plot_model_comparison(results, output_paths["figures"] / "model_comparison.png")
    plot_abstention_comparison(abstention_tables, output_paths["figures"] / "abstention_comparison.png")

    return {
        "output_dir": str(output_dir),
        "results": results,
        "metadata": metadata,
    }
