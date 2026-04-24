from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nordic_mushrooms import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet-based mushroom species models on DF20-style metadata."
    )
    parser.add_argument("--metadata-path", default=None, help="Single metadata table, optionally with a split column.")
    parser.add_argument("--train-metadata-path", default=None, help="Train metadata file if split files are separate.")
    parser.add_argument("--val-metadata-path", default=None, help="Optional validation metadata file.")
    parser.add_argument("--test-metadata-path", default=None, help="Test metadata file if split files are separate.")
    parser.add_argument("--images-root", default=None, help="Root folder containing the mushroom images.")
    parser.add_argument("--path-column", default=None, help="Override the detected image path column.")
    parser.add_argument("--species-column", default=None, help="Override the detected species column.")
    parser.add_argument("--split-column", default=None, help="Override the detected split column.")
    parser.add_argument("--risk-map-path", default="data/risk_map.csv")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--top-species", type=int, default=100)
    parser.add_argument("--min-images-per-species", type=int, default=30)
    parser.add_argument("--max-images-per-species", type=int, default=None)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output-dir", default="outputs/df20_species_project")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_config = ExperimentConfig()
    config = ExperimentConfig(
        metadata_path=args.metadata_path,
        train_metadata_path=args.train_metadata_path,
        val_metadata_path=args.val_metadata_path,
        test_metadata_path=args.test_metadata_path,
        images_root=args.images_root,
        path_column=args.path_column,
        species_column=args.species_column,
        split_column=args.split_column,
        risk_map_path=args.risk_map_path,
        model_names=tuple(args.models) if args.models else default_config.model_names,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        num_workers=args.num_workers,
        top_species=args.top_species if args.top_species and args.top_species > 0 else None,
        min_images_per_species=args.min_images_per_species,
        max_images_per_species=args.max_images_per_species,
        output_dir=args.output_dir,
        pretrained=not args.no_pretrained,
        label_smoothing=args.label_smoothing,
        val_size=args.val_size,
        test_size=args.test_size,
        use_class_weights=args.use_class_weights,
        use_weighted_sampler=not args.no_weighted_sampler,
    )
    bundle = run_experiment(config)
    print(f"Saved results to: {bundle['output_dir']}")


if __name__ == "__main__":
    main()
