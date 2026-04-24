from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mushroom_risk import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mushroom risk classification models.")
    parser.add_argument("--data-dir", default=None, help="Optional local dataset directory.")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--min-images-per-class", type=int, default=3)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--output-dir", default="outputs/mushroom_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_config = ExperimentConfig()
    config = ExperimentConfig(
        model_names=tuple(args.models) if args.models else default_config.model_names,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        num_workers=args.num_workers,
        min_images_per_class=args.min_images_per_class,
        max_images_per_class=args.max_images_per_class,
        output_dir=args.output_dir,
        label_smoothing=args.label_smoothing,
        use_class_weights=args.use_class_weights,
        use_weighted_sampler=not args.no_weighted_sampler,
    )
    bundle = run_experiment(config, data_dir=args.data_dir)
    print(f"Saved results to: {bundle['output_dir']}")


if __name__ == "__main__":
    main()
