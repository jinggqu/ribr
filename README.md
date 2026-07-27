# RIBR Ultrasound Segmentation

[[Paper](http://arxiv.org/abs/2607.21787)] [[Datasets](https://github.com/jinggqu/ribr/releases)]

Minimal training and evaluation code for **Risk-Routed Implicit Boundary
Refinement for Robust Ultrasound Image Segmentation**.

## Environment

Requirements: Python 3.12+, `uv`, and Bash 3.2+. A CUDA GPU is recommended for
training; the smoke test runs on CPU.

```bash
uv sync --frozen
uv run python -m src.jobs.smoke_test
```

The smoke test creates temporary synthetic data, trains for one epoch, and
tests the resulting checkpoint.

## Data

Download `ribr-data-v1.tar.gz` from the
[`data-v1` release](https://github.com/jinggqu/ribr/releases/tag/data-v1) and
extract it. The archive contains 256×256 images and masks for BUS-UCLM, BUSI,
Breast-Lesions-USG, DDTI, LymphUS-Center-1, LymphUS-Center-2, Prostate, and
TN3K. LN-INT and LN-EXT are private hospital datasets and they are not publicly
available due to institutional data sharing agreements.

```text
ribr-data-v1/
├── catalog/samples-v1.csv
├── derivatives/resize256-v1/
│   ├── manifest.csv
│   └── <dataset>/
│       ├── images/
│       └── masks/
├── data_views/ribr-v1.csv
└── data_splits/ribr-v1/
    ├── seed-1.csv
    ├── seed-2.csv
    └── seed-3.csv
```

Configuration:

```bash
export DATASET=BUSI
export DATA_ROOT=/path/to/ribr-data-v1
export DATA_VIEW="$DATA_ROOT/data_views/ribr-v1.csv"
export SPLIT_ROOT="$DATA_ROOT/data_splits/ribr-v1"
export DERIVATIVE_MANIFEST="$DATA_ROOT/derivatives/resize256-v1/manifest.csv"
```

## Train and test

This command trains RIBR and then evaluates the best checkpoint on the test
split:

```bash
uv run python -m src.methods.ribr.train_seg \
  --exp example-ribr-1 \
  --dataset "$DATASET" \
  --data-root "$DATA_ROOT" \
  --data-view "$DATA_VIEW" \
  --split-root "$SPLIT_ROOT" \
  --derivative-manifest "$DERIVATIVE_MANIFEST" \
  --img-size 256 \
  --batch-size 1 \
  --num-workers 0 \
  --seed 1 \
  --device cuda:0 \
  --weak-augs
```

Main outputs:

```text
runs/example-ribr-1/ExampleUS/256/train/best_model.pt
runs/example-ribr-1/ExampleUS/256/test/<run>/results.csv
runs/example-ribr-1/ExampleUS/256/test/<run>/per_sample.csv
```

To test an existing checkpoint, run the same command with:

```text
--test --checkpoint /path/to/best_model.pt
```

Add `--test-all-splits` only when evaluation on train, validation, and test
samples is intended.

## Fixed experiment script

Create `seed-1.csv`, `seed-2.csv`, and `seed-3.csv`, then run:

```bash
DATA_ROOT="$DATA_ROOT" \
DATA_VIEW="$DATA_VIEW" \
SPLIT_ROOT="$SPLIT_ROOT" \
DERIVATIVE_MANIFEST="$DERIVATIVE_MANIFEST" \
TRAIN_DATASETS="$DATASET" \
SEEDS="1 2 3" \
DEVICE=cuda:0 \
bash scripts/segmentation/run_ribr_experiments.sh
```

The script skips completed result directories.

## License

CC BY-NC 4.0. See `LICENSE`.
