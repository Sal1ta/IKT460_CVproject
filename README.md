# Mushroom Risk Classification with ResNet and Successor Models

This project is a clean rebuild of the course experiment around one task only: **4-class mushroom risk classification**. The baseline model is **ResNet-50** from *Deep Residual Learning for Image Recognition* (CVPR 2016 Best Paper), and it is compared against three later architectures:

- `resnet50` -> baseline
- `resnext50_32x4d` -> successor architecture with aggregated residual transformations
- `densenet121` -> later high-performing CNN that cites and outperforms ResNet on standard benchmarks
- `seresnet50` -> ResNet with squeeze-and-excitation attention

The task is intentionally framed as a **research prototype**, not a real safety tool. The key question is whether later architectures improve mushroom risk recognition while also reducing **dangerous mistakes**, where unsafe mushrooms are predicted as safe.

## Research Question

**Which fine-tuned architecture performs best on 4-class mushroom risk classification, and does a stronger architecture lower the dangerous error rate on poisonous and deadly mushrooms compared with the original ResNet baseline?**

## Project Structure

```text
CVproject-rebuild/
  notebooks/
    mushroom_risk_comparison.ipynb
  scripts/
    train_mushroom_models.py
  src/
    mushroom_risk/
      __init__.py
      data.py
      models.py
      training.py
      utils.py
  README.md
  requirements.txt
```

## Dataset

The experiment uses the Kaggle dataset:

`derekkunowilliams/mushrooms`

The loader:

- downloads through `kagglehub` if needed
- reuses the local cache if the dataset is already present
- infers the 4 risk labels from folder names:
  - `edible`
  - `conditionally_edible`
  - `poisonous`
  - `deadly`

## Metrics

The comparison focuses on:

- test accuracy
- balanced accuracy
- macro F1
- dangerous error rate

`dangerous_error_rate` is the fraction of truly unsafe mushrooms (`poisonous` or `deadly`) that were incorrectly predicted as safe (`edible` or `conditionally_edible`).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the Four Models

```bash
python3 scripts/train_mushroom_models.py \
  --models resnet50 resnext50_32x4d densenet121 seresnet50 \
  --epochs 12 \
  --batch-size 8 \
  --output-dir outputs/mushroom_comparison
```

For a quick smoke test:

```bash
python3 scripts/train_mushroom_models.py \
  --models resnet50 \
  --epochs 1 \
  --batch-size 2 \
  --max-images-per-class 4 \
  --output-dir outputs/smoke
```

## Notebook Workflow

Open `notebooks/mushroom_risk_comparison.ipynb`.

The notebook is set up to:

- load finished results by default
- summarize metrics from `outputs/mushroom_comparison/results.csv`
- display the comparison plot
- display per-class metrics
- display confusion matrices for each model

Set `RUN_TRAINING = True` only if you deliberately want to launch training from inside the notebook.

## Output Files

Each run saves:

- `metadata.json`
- `results.csv`
- `model_comparison.png`
- one checkpoint per model
- one training curve per model
- one confusion matrix per model
- one per-class metrics CSV per model

## Safety Note

This repository is for coursework and research only. Mushroom identification is safety-critical, and the predictions produced here must **not** be used to decide whether a mushroom is safe to eat.
