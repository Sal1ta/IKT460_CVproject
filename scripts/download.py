"""
DF20 dataset setup helper.

Downloads the DF20 metadata zip and extracts the CSV files.
Images must be downloaded separately (~6.5 GB).

Usage:
    python scripts/setup_df20.py --data-dir data/df20
    python scripts/setup_df20.py --data-dir data/df20 --verify-images
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "http://ptak.felk.cvut.cz/plants/DanishFungiDataset"

METADATA_ZIP_URL = f"{BASE_URL}/DF20-metadata.zip"
IMAGES_URL       = f"{BASE_URL}/DF20-300px.tar.gz"


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, 100 * downloaded / total_size)
        mb = downloaded / 1_048_576
        print(f"\r  {pct:5.1f}%  {mb:.1f} MB", end="", flush=True)


def download_metadata(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "DF20-metadata.zip"

    if not zip_path.exists():
        print(f"Downloading DF20 metadata zip ...")
        print(f"  from: {METADATA_ZIP_URL}")
        try:
            urllib.request.urlretrieve(METADATA_ZIP_URL, zip_path, reporthook=_progress_hook)
            print()
        except Exception as exc:
            print(f"\n  FAILED: {exc}")
            print(f"  Download manually from: {METADATA_ZIP_URL}")
            print(f"  and place the zip at: {zip_path}")
            return
    else:
        print(f"  [skip] metadata zip already exists")

    print(f"Extracting {zip_path.name} ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        print(f"  Extracted to {data_dir}")
    except Exception as exc:
        print(f"  Extraction failed: {exc}")


def find_metadata_csvs(data_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    candidates = {
        "train": ["DF20_train_metadata.csv", "train_metadata.csv"],
        "val":   ["DF20_val_metadata.csv",   "val_metadata.csv"],
        "test":  ["DF20_test_metadata.csv",  "test_metadata.csv"],
    }
    for split, names in candidates.items():
        for name in names:
            for path in data_dir.rglob(name):
                found[split] = path
                break
    return found


def verify_metadata(data_dir: Path) -> dict[str, Path]:
    csvs = find_metadata_csvs(data_dir)
    print("\nMetadata files:")
    for split in ("train", "val", "test"):
        if split in csvs:
            rows = sum(1 for _ in open(csvs[split], encoding="utf-8")) - 1
            print(f"  [OK]      {split:5s} — {csvs[split].name} ({rows:,} rows)")
        else:
            print(f"  [MISSING] {split}")
    return csvs


def verify_images(data_dir: Path) -> None:
    image_exts = {".jpg", ".jpeg", ".png"}
    candidates = list(data_dir.rglob("images")) + list(data_dir.rglob("images_300px"))
    images_root = candidates[0] if candidates else None

    if images_root and images_root.is_dir():
        count = sum(1 for p in images_root.rglob("*") if p.suffix.lower() in image_exts)
        print(f"\nImages: {images_root}")
        print(f"  {count:,} image files found")
        if count < 1000:
            print("  [WARNING] Very few images — download may be incomplete.")
    else:
        print(f"\n  [MISSING] No images folder found under {data_dir}")
        print(f"  Download images from: {IMAGES_URL}")


def print_next_steps(data_dir: Path, csvs: dict[str, Path]) -> None:
    train = csvs.get("train", data_dir / "DF20_train_metadata.csv")
    val   = csvs.get("val",   data_dir / "DF20_val_metadata.csv")
    test  = csvs.get("test",  data_dir / "DF20_test_metadata.csv")

    print()
    print("=" * 65)
    print("NEXT STEPS")
    print("=" * 65)
    print()
    if "train" not in csvs:
        print("1. Download images + metadata:")
        print(f"   Images (~6.5 GB): {IMAGES_URL}")
        print(f"   Extract into:     {data_dir}/")
        print()
    else:
        print("1. Download images (~6.5 GB):")
        print(f"   {IMAGES_URL}")
        print()
        print("   Using curl:")
        print(f"   curl -O --output-dir {data_dir} {IMAGES_URL}")
        print(f"   tar -xzf {data_dir}/DF20-300px.tar.gz -C {data_dir}/")
        print()

    print("2. Run training:")
    print()
    print("   python scripts/train_df20_models.py \\")
    print(f"       --train-metadata-path \"{train}\" \\")
    print(f"       --val-metadata-path   \"{val}\" \\")
    print(f"       --test-metadata-path  \"{test}\" \\")
    print(f"       --images-root         \"{data_dir}/images\" \\")
    print("       --models resnet50 resnext50_32x4d seresnet50 convnext_tiny \\")
    print("       --top-species 100 --min-images-per-species 30 \\")
    print("       --epochs 15 --batch-size 32 \\")
    print("       --output-dir outputs/df20_species_project")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DF20 dataset setup helper.")
    parser.add_argument("--data-dir", default="data/df20")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--verify-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    print(f"DF20 data directory: {data_dir}\n")

    if not args.skip_download:
        download_metadata(data_dir)

    csvs = verify_metadata(data_dir)

    if args.verify_images:
        verify_images(data_dir)

    print_next_steps(data_dir, csvs)


if __name__ == "__main__":
    main()
