from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from .risk import normalize_species_key

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SpeciesSample:
    path: Path
    species_key: str
    species_name: str
    split: str | None = None
    observation_id: str | None = None
    month: str | None = None
    habitat: str | None = None
    substrate: str | None = None
    genus: str | None = None
    family: str | None = None


class SpeciesClassificationDataset(Dataset):
    def __init__(
        self,
        samples: list[SpeciesSample],
        label_to_index: dict[str, int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.samples = samples
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label_to_index[sample.species_key], index


def normalize_column_name(value: str) -> str:
    return "".join(character for character in value.strip().lower() if character.isalnum())


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", keep_default_na=False)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python", keep_default_na=False)
    raise ValueError(f"Unsupported metadata file format: {path.suffix}")


def choose_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
    explicit: str | None = None,
) -> str | None:
    if explicit:
        if explicit not in dataframe.columns:
            raise ValueError(f"Column '{explicit}' was not found in metadata.")
        return explicit

    normalized_to_original = {
        normalize_column_name(column): column for column in dataframe.columns
    }
    for candidate in candidates:
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]
    return None


def resolve_image_path(
    image_value: str,
    metadata_path: Path,
    images_root: Path | None,
) -> Path:
    candidate = Path(str(image_value))
    if candidate.is_absolute():
        return candidate

    candidate_bases = [base for base in (images_root, metadata_path.parent, metadata_path.parent / "images") if base]
    for base in candidate_bases:
        resolved = base / candidate
        if resolved.exists():
            return resolved
    if images_root is not None:
        return images_root / candidate
    return metadata_path.parent / candidate


def build_samples_from_table(
    metadata_path: str | Path,
    images_root: str | Path | None = None,
    path_column: str | None = None,
    species_column: str | None = None,
    split_column: str | None = None,
    default_split: str | None = None,
) -> tuple[list[SpeciesSample], dict[str, int]]:
    table_path = Path(metadata_path)
    images_root_path = Path(images_root) if images_root else None
    dataframe = _read_table(table_path)

    image_col = choose_column(
        dataframe,
        (
            "imagepath",
            "filepath",
            "path",
            "filename",
            "imagename",
            "imgpath",
            "file",
            "image",
            "relativepath",
        ),
        explicit=path_column,
    )
    species_col = choose_column(
        dataframe,
        (
            "species",
            "speciesname",
            "scientificname",
            "taxonname",
            "classname",
            "label",
            "labelname",
            "taxon",
        ),
        explicit=species_column,
    )
    split_col = choose_column(
        dataframe,
        (
            "split",
            "subset",
            "partition",
            "fold",
            "datasetpart",
        ),
        explicit=split_column,
    )
    observation_col = choose_column(
        dataframe,
        (
            "observationid",
            "observation",
            "obsid",
        ),
    )
    month_col = choose_column(dataframe, ("month",))
    habitat_col = choose_column(dataframe, ("habitat",))
    substrate_col = choose_column(dataframe, ("substrate",))
    genus_col = choose_column(dataframe, ("genus",))
    family_col = choose_column(dataframe, ("family",))

    if image_col is None or species_col is None:
        raise ValueError(
            f"Metadata at {table_path} must contain image path and species columns. "
            "Use --path-column and --species-column if auto-detection fails."
        )

    samples: list[SpeciesSample] = []
    stats = {"rows": len(dataframe), "missing_paths": 0, "empty_species": 0}

    for row_dict in dataframe.to_dict(orient="records"):
        species_name = str(row_dict[species_col]).strip()
        if not species_name:
            stats["empty_species"] += 1
            continue

        image_path = resolve_image_path(row_dict[image_col], table_path, images_root_path)
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            image_path = image_path.with_suffix(".jpg") if not image_path.suffix else image_path
        if not image_path.exists():
            stats["missing_paths"] += 1
            continue

        samples.append(
            SpeciesSample(
                path=image_path,
                species_key=normalize_species_key(species_name),
                species_name=species_name,
                split=str(row_dict[split_col]).strip().lower() if split_col and str(row_dict[split_col]).strip() else default_split,
                observation_id=str(row_dict[observation_col]).strip() if observation_col and str(row_dict[observation_col]).strip() else None,
                month=str(row_dict[month_col]).strip() if month_col and str(row_dict[month_col]).strip() else None,
                habitat=str(row_dict[habitat_col]).strip() if habitat_col and str(row_dict[habitat_col]).strip() else None,
                substrate=str(row_dict[substrate_col]).strip() if substrate_col and str(row_dict[substrate_col]).strip() else None,
                genus=str(row_dict[genus_col]).strip() if genus_col and str(row_dict[genus_col]).strip() else None,
                family=str(row_dict[family_col]).strip() if family_col and str(row_dict[family_col]).strip() else None,
            )
        )

    return samples, stats


def load_species_samples(
    metadata_path: str | None = None,
    train_metadata_path: str | None = None,
    val_metadata_path: str | None = None,
    test_metadata_path: str | None = None,
    images_root: str | None = None,
    path_column: str | None = None,
    species_column: str | None = None,
    split_column: str | None = None,
) -> tuple[list[SpeciesSample], dict[str, dict[str, int]]]:
    stats: dict[str, dict[str, int]] = {}
    all_samples: list[SpeciesSample] = []

    if metadata_path:
        combined_samples, combined_stats = build_samples_from_table(
            metadata_path,
            images_root=images_root,
            path_column=path_column,
            species_column=species_column,
            split_column=split_column,
            default_split=None,
        )
        stats["combined"] = combined_stats
        all_samples.extend(combined_samples)
    else:
        for name, metadata_file in (
            ("train", train_metadata_path),
            ("val", val_metadata_path),
            ("test", test_metadata_path),
        ):
            if not metadata_file:
                continue
            split_samples, split_stats = build_samples_from_table(
                metadata_file,
                images_root=images_root,
                path_column=path_column,
                species_column=species_column,
                split_column=split_column,
                default_split=name,
            )
            stats[name] = split_stats
            all_samples.extend(split_samples)

    if not all_samples:
        raise ValueError(
            "No samples were loaded. Provide --metadata-path or train/test metadata paths."
        )

    return all_samples, stats


def stratified_split(
    samples: list[SpeciesSample],
    seed: int,
    val_size: float,
    test_size: float,
) -> dict[str, list[SpeciesSample]]:
    labels = [sample.species_key for sample in samples]
    can_stratify = min(Counter(labels).values()) >= 2

    train_samples, temp_samples, _, temp_labels = train_test_split(
        samples,
        labels,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=labels if can_stratify else None,
    )

    relative_test_size = test_size / (val_size + test_size)
    temp_counts = Counter(temp_labels)
    can_stratify_temp = min(temp_counts.values()) >= 2 if temp_counts else False

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_labels if can_stratify_temp else None,
    )
    return {"train": train_samples, "val": val_samples, "test": test_samples}


def build_splits(
    samples: list[SpeciesSample],
    seed: int,
    val_size: float,
    test_size: float,
) -> tuple[dict[str, list[SpeciesSample]], str]:
    split_groups: dict[str, list[SpeciesSample]] = {"train": [], "val": [], "test": []}

    for sample in samples:
        split_value = (sample.split or "").lower()
        if split_value in {"train", "training"}:
            split_groups["train"].append(sample)
        elif split_value in {"val", "valid", "validation", "dev"}:
            split_groups["val"].append(sample)
        elif split_value in {"test", "testing"}:
            split_groups["test"].append(sample)

    if not split_groups["train"] and not split_groups["test"]:
        return stratified_split(samples, seed, val_size, test_size), "stratified_from_single_table"

    if not split_groups["train"] or not split_groups["test"]:
        raise ValueError(
            "Metadata-based split detection found incomplete split labels. "
            "Use train/test metadata files or provide a metadata table with train/test split values."
        )

    if not split_groups["val"]:
        train_samples = split_groups["train"]
        train_labels = [sample.species_key for sample in train_samples]
        can_stratify = min(Counter(train_labels).values()) >= 2
        split_groups["train"], split_groups["val"] = train_test_split(
            train_samples,
            test_size=val_size,
            random_state=seed,
            stratify=train_labels if can_stratify else None,
        )
        strategy = "provided_train_test_plus_generated_val"
    else:
        strategy = "provided_train_val_test"

    return split_groups, strategy


def _sample_limit_per_species(
    samples: list[SpeciesSample],
    max_images_per_species: int | None,
    seed: int,
    split_name: str,
) -> list[SpeciesSample]:
    if max_images_per_species is None:
        return samples

    grouped: dict[str, list[SpeciesSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.species_key, []).append(sample)

    rng = random.Random(seed + sum(ord(character) for character in split_name))
    limited: list[SpeciesSample] = []
    for species_key in sorted(grouped):
        species_samples = grouped[species_key][:]
        rng.shuffle(species_samples)
        limited.extend(species_samples[:max_images_per_species])
    return limited


def filter_splits_to_species_subset(
    splits: dict[str, list[SpeciesSample]],
    top_species: int | None,
    min_images_per_species: int,
    max_images_per_species: int | None,
    seed: int,
) -> tuple[dict[str, list[SpeciesSample]], dict[str, int]]:
    train_counts = Counter(sample.species_key for sample in splits["train"])
    eligible = [(species, count) for species, count in train_counts.items() if count >= min_images_per_species]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    if top_species is not None:
        eligible = eligible[:top_species]

    selected_species = {species for species, _ in eligible}
    if len(selected_species) < 2:
        raise ValueError(
            "Filtering left fewer than two species. Lower --min-images-per-species or increase --top-species."
        )

    filtered = {
        split_name: _sample_limit_per_species(
            [sample for sample in split_samples if sample.species_key in selected_species],
            max_images_per_species,
            seed,
            split_name,
        )
        for split_name, split_samples in splits.items()
    }

    if not filtered["val"] or not filtered["test"]:
        raise ValueError(
            "The selected species subset is missing validation or test samples. "
            "Try lowering --min-images-per-species or using a larger top-species value."
        )

    selected_counts = {species: train_counts[species] for species in selected_species}
    return filtered, dict(sorted(selected_counts.items(), key=lambda item: (-item[1], item[0])))


def build_label_map(
    train_samples: list[SpeciesSample],
) -> tuple[list[str], dict[str, str], dict[str, int]]:
    species_order = sorted(
        Counter(sample.species_key for sample in train_samples).items(),
        key=lambda item: (-item[1], item[0]),
    )
    class_keys = [species for species, _ in species_order]
    display_lookup = {}
    for sample in train_samples:
        display_lookup.setdefault(sample.species_key, sample.species_name)
    label_to_index = {species: index for index, species in enumerate(class_keys)}
    return class_keys, display_lookup, label_to_index


def get_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_transform, eval_transform
