from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import kagglehub
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RISK_ORDER = ["edible", "conditionally_edible", "poisonous", "deadly"]
SAFE_RISKS = {"edible", "conditionally_edible"}
UNSAFE_RISKS = {"poisonous", "deadly"}


@dataclass(frozen=True)
class MushroomSample:
    path: Path
    species: str
    risk: str


class MushroomRiskDataset(Dataset):
    def __init__(
        self,
        samples: list[MushroomSample],
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
        return image, self.label_to_index[sample.risk]


def normalize_name(value: str) -> str:
    cleaned = value.strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def canonical_risk(value: str) -> str | None:
    aliases = {
        "edible": "edible",
        "eatable": "edible",
        "conditionally_edible": "conditionally_edible",
        "conditionnally_edible": "conditionally_edible",
        "conditional_edible": "conditionally_edible",
        "not_recommended": "conditionally_edible",
        "poisonous": "poisonous",
        "poison": "poisonous",
        "toxic": "poisonous",
        "inedible": "poisonous",
        "non_edible": "poisonous",
        "not_edible": "poisonous",
        "deadly": "deadly",
        "deadly_poisonous": "deadly",
        "lethal": "deadly",
    }
    return aliases.get(normalize_name(value))


def download_dataset(dataset_id: str) -> Path:
    owner, dataset_name = dataset_id.split("/", maxsplit=1)
    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets" / owner / dataset_name / "versions"
    if cache_root.exists():
        cached_versions = sorted(
            [path for path in cache_root.iterdir() if path.is_dir()],
            key=lambda path: path.name,
        )
        if cached_versions:
            return cached_versions[-1]
    return Path(kagglehub.dataset_download(dataset_id))


def find_image_root(base_dir: Path) -> Path:
    image_counts: dict[Path, int] = {}
    for image_path in base_dir.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            for parent in image_path.parents:
                if parent == base_dir or base_dir in parent.parents:
                    image_counts[parent] = image_counts.get(parent, 0) + 1
    if not image_counts:
        raise FileNotFoundError(f"No image files found under {base_dir}")
    return max(image_counts, key=image_counts.get)


def infer_sample(root: Path, image_path: Path) -> MushroomSample | None:
    parts = list(image_path.relative_to(root).parts[:-1])
    normalized_parts = [normalize_name(part) for part in parts]

    risk = None
    risk_index = None
    for index, part in enumerate(normalized_parts):
        maybe_risk = canonical_risk(part)
        if maybe_risk:
            risk = maybe_risk
            risk_index = index
            break

    if risk is None or risk_index is None:
        return None

    if risk_index + 1 < len(parts):
        species = normalize_name(parts[risk_index + 1])
    elif parts:
        species = normalize_name(parts[-1])
    else:
        species = normalize_name(image_path.stem)

    if not species or species in set(RISK_ORDER):
        species = normalize_name(image_path.stem)

    return MushroomSample(path=image_path, species=species, risk=risk)


def load_samples(dataset_id: str, data_dir: str | None = None) -> tuple[Path, list[MushroomSample]]:
    base_dir = Path(data_dir) if data_dir else download_dataset(dataset_id)
    root = find_image_root(base_dir)
    samples = [
        sample
        for image_path in root.rglob("*")
        if image_path.is_file()
        and image_path.suffix.lower() in IMAGE_EXTENSIONS
        and (sample := infer_sample(root, image_path)) is not None
    ]
    if not samples:
        raise ValueError(f"No recognizable mushroom samples found under {root}")
    return root, samples


def filter_samples(
    samples: list[MushroomSample],
    min_images_per_class: int,
    max_images_per_class: int | None,
    seed: int,
) -> list[MushroomSample]:
    grouped: dict[str, list[MushroomSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.risk].append(sample)

    rng = random.Random(seed)
    filtered: list[MushroomSample] = []
    for risk, risk_samples in grouped.items():
        if len(risk_samples) < min_images_per_class:
            continue
        rng.shuffle(risk_samples)
        if max_images_per_class is not None:
            risk_samples = risk_samples[:max_images_per_class]
        filtered.extend(risk_samples)

    present_classes = {sample.risk for sample in filtered}
    if len(present_classes) < 2:
        raise ValueError("Not enough classes after filtering.")

    rng.shuffle(filtered)
    return filtered


def split_samples(
    samples: list[MushroomSample],
    seed: int,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> dict[str, list[MushroomSample]]:
    labels = [sample.risk for sample in samples]
    can_stratify = min(Counter(labels).values()) >= 2

    train_samples, temp_samples, _, temp_labels = train_test_split(
        samples,
        labels,
        test_size=(val_size + test_size),
        random_state=seed,
        stratify=labels if can_stratify else None,
    )

    temp_counts = Counter(temp_labels)
    can_stratify_temp = min(temp_counts.values()) >= 2 if temp_counts else False
    relative_test_size = test_size / (val_size + test_size)

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_labels if can_stratify_temp else None,
    )

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }


def build_label_map(samples: list[MushroomSample]) -> tuple[list[str], dict[str, int]]:
    labels = [risk for risk in RISK_ORDER if any(sample.risk == risk for sample in samples)]
    return labels, {label: index for index, label in enumerate(labels)}


def get_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
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
