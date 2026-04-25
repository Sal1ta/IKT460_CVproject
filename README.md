# Risk-Aware Nordic Mushroom Recognition with ResNet and Successor Models

This repository contains the coursework implementation for a **species-first mushroom recognition project** built around the CVPR 2016 Best Paper **ResNet** baseline and three later CNN architectures:

- `resnet50` -> **Deep Residual Learning for Image Recognition** (baseline, CVPR 2016 Best Paper)
- `resnext50_32x4d` -> **Aggregated Residual Transformations for Deep Neural Networks** (CVPR 2017)
- `seresnet50` -> **Squeeze-and-Excitation Networks** (CVPR 2018 Best Paper)
- `convnext_tiny` -> **A ConvNet for the 2020s** (CVPR 2022)

Instead of training on coarse folder labels such as “edible” or “poisonous”, the project is framed as:

1. **Recognize Nordic mushroom species from field images**
2. **Map species predictions to safety risk labels**
3. **Measure dangerous confusions and confidence-based abstention**

That makes the project more faithful to real foraging situations in Scandinavia, where the real challenge is often species confusion, not binary safety labels.

## Research Question

**How well do ResNet-based architectures recognize Nordic wild mushroom species from field images, and can confidence-based abstention reduce dangerous confusions between edible and poisonous species?**

## Why This Matters

The Norwegian Poisons Information Centre warns that poisonous mushrooms in Norway can be confused with edible species and that people should only eat mushrooms they are **100% sure** are safe. This makes risk-aware visual recognition a relevant computer vision problem for Norwegian and Scandinavian society, even if the final system is strictly a research prototype and not a real safety tool.

## Dataset Choice

The intended dataset family is **Danish Fungi 2020 (DF20)**, a taxonomy-accurate fine-grained benchmark built from the Atlas of Danish Fungi. The code in this repository is written to support:

- a **single metadata file** with a split column
- or **separate train/test metadata files**
- optional validation metadata
- local image folders downloaded from the official DF20/DF24 repository

The training pipeline is metadata-driven rather than hardcoded to one folder structure, so it can support the official DF20 release, the newer DF24 split, or a course subset built from the same metadata format.

## Main Project Idea

The main experiment compares four ImageNet-pretrained CNNs on a **species classification task**. Risk is then used as an **analysis layer**:

- species classification is the supervised task
- species predictions are mapped to `edible`, `conditionally_edible`, `poisonous`, `deadly`, or `unknown`
- dangerous mistakes are counted when a truly unsafe species is predicted as safe
- an abstention analysis checks whether requiring higher confidence reduces dangerous errors

## Project Structure

```text
IKT460_CVproject/
  data/
    risk_map.csv          ← species-to-risk lookup (94 species)
    df20/                 ← DF20 metadata CSVs + images (gitignored)
  nordic_mushrooms/       ← core Python package
    __init__.py
    data.py
    models.py
    risk.py
    training.py
    utils.py
  notebooks/
    analysis.ipynb        ← main analysis notebook
  outputs/                ← training results (gitignored)
  scripts/
    train.py              ← CLI entry point for training
    download.py           ← DF20 dataset setup helper
  README.md
  requirements.txt
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected Data Inputs

You need:

- the DF20 or DF24 image files locally
- at least one metadata table with:
  - an image path column
  - a species/scientific-name column
  - optionally a split column

If the official dataset gives you **separate train/test metadata files**, use those directly.

## Example Training Commands

### Simple default command

If the DF20 files are in the expected folder:

```text
data/df20/DF20-train_metadata_PROD-2.csv
data/df20/DF20-public_test_metadata_PROD-2.csv
data/df20/DF20_300/
```

then the normal 4-model comparison can be started with:

```bash
python run.py
```

For a faster check with only ResNet-50:

```bash
python run.py --preset quick
```

For a longer run:

```bash
python run.py --preset full
```

If you manually ask for 15 epochs, the model will now run those epochs unless it reaches 100% training accuracy. Patience-based early stopping is only shortened when you explicitly pass `--patience`.

On JupyterHub, run it in the background with:

```bash
nohup python run.py > training.log 2>&1 &
tail -f training.log
```

### 1. One metadata file with a split column

```bash
python scripts/train.py \
  --metadata-path /path/to/df20_metadata.csv \
  --images-root /path/to/df20_images_300px \
  --models resnet50 resnext50_32x4d seresnet50 convnext_tiny \
  --top-species 100 \
  --min-images-per-species 30 \
  --epochs 15 \
  --batch-size 16 \
  --output-dir outputs/df20_species_project
```

### 2. Separate train/test metadata files

```bash
python scripts/train.py \
  --train-metadata-path /path/to/train_metadata.csv \
  --test-metadata-path /path/to/test_metadata.csv \
  --images-root /path/to/df20_images_300px \
  --models resnet50 resnext50_32x4d seresnet50 convnext_tiny \
  --top-species 100 \
  --min-images-per-species 30 \
  --epochs 15 \
  --batch-size 16 \
  --output-dir outputs/df20_species_project
```

### 3. Fast smoke test

```bash
python scripts/train.py \
  --metadata-path /path/to/metadata.csv \
  --images-root /path/to/images \
  --models resnet50 \
  --top-species 4 \
  --min-images-per-species 2 \
  --max-images-per-species 4 \
  --epochs 1 \
  --batch-size 2 \
  --output-dir outputs/smoke_df20
```

## Metrics

The saved comparison focuses on:

- top-1 accuracy
- top-3 accuracy
- balanced accuracy
- macro F1
- risk coverage
- risk accuracy
- dangerous error rate

`dangerous_error_rate` is the fraction of truly unsafe mushrooms (`poisonous` or `deadly`) that are predicted as safe (`edible` or `conditionally_edible`).

## Saved Outputs

Each full run saves:

- `metadata.json`

```text
outputs/df20_species_project/
  metadata.json
  figures/
    model_comparison.png
    abstention_comparison.png
    <model>_history.png
    <model>_risk_confusion.png
  tables/
    results.csv
    <model>_per_class_metrics.csv
    <model>_top_confusions.csv
    <model>_abstention.csv
  predictions/
    <model>_predictions.csv
  checkpoints/
    <model>_best.pt
```

## Notebook Workflow

Open `notebooks/analysis.ipynb`.

The notebook is designed to:

- load finished results by default
- summarize `tables/results.csv`
- display comparison graphs
- display training curves
- display risk confusion matrices
- build a readable species confusion matrix for the best model
- compare abstention behaviour across models
- inspect the most important risky confusions

## Important Modeling Choices

- **Species-first supervision**: the model learns species labels, not risk folders
- **Risk-aware evaluation**: safety is analyzed after prediction using a curated species-to-risk map
- **Weighted sampling**: enabled by default to soften the long-tail distribution
- **Confidence abstention**: included to support a “defer to expert” safety story
- **Top-species subset**: enabled so the course experiment can stay tractable while still using a real DF20-style benchmark

## Safety Note

This repository is for coursework and research only. It must **not** be used to decide whether a mushroom is safe to eat.

## Source Links

- Official CVPR awards list: [IEEE CVPR Paper Awards](https://tc.computer.org/tcpami/awards/cvpr-paper-awards/)
- ResNet paper: [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
- DF20 paper: [Danish Fungi 2020 - Not Just Another Image Recognition Dataset](https://researchprofiles.ku.dk/en/publications/danish-fungi-2020-not-just-another-image-recognition-dataset)
- DF20 dataset/code repository: [BohemianVRA/DanishFungiDataset](https://github.com/BohemianVRA/DanishFungiDataset)
- Norwegian mushroom safety guidance: [Poisonous mushrooms in Norway - Helsenorge](https://www.helsenorge.no/en/poison-information/poisonous-mushrooms/)
