<div align="center">

# FGL-SAN
### Fuzzy Graph Laplacian-Regularized Shallow Attention Network

**T-cell biomarker discovery from single-cell / spatial transcriptomics data**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## Overview

FGL-SAN is a shallow attention network for single-cell transcriptomics classification and
biomarker discovery. It learns a sparse, per-gene attention vector that both drives
classification and doubles as a ranked biomarker signal — regularized by the geometry of
the data itself via a fuzzy k-nearest-neighbor graph Laplacian.

## Table of Contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Outputs](#outputs)
- [Known limitations](#known-limitations)

## How it works

```
 Raw counts (.h5ad)
        │
        ▼
 ┌──────────────────────┐
 │ Preprocessing          │  filter → CPM normalize → log1p → scale
 └──────────┬─────────────┘
            ▼
 ┌──────────────────────┐
 │ Fuzzy kNN graph        │  adaptive Gaussian bandwidth per cell →
 │ + normalized Laplacian │  symmetrize → L_sym
 └──────────┬─────────────┘
            ▼
 ┌──────────────────────┐
 │ Shallow attention net  │  attention(θ) ⊙ X → hidden layer → softmax
 └──────────┬─────────────┘
            ▼
 ┌────────────────────────────────────────────────┐
 │ Loss = CrossEntropy + γ·Tr(Ŷᵀ L_sym Ŷ) + λ‖θ‖₁  │
 └──────────┬───────────────────────────────────────┘
            ▼
 Metrics (ARI / NMI / F1 / ROC-AUC)  +  Ranked biomarker list
```

| Stage | Description |
|---|---|
| **Preprocessing** | Filter cells/genes, CPM-normalize, log1p, scale (`scanpy`) |
| **Fuzzy graph construction** | kNN graph with adaptive (self-tuning) Gaussian bandwidths per cell, symmetrized, converted to the normalized graph Laplacian `L_sym` |
| **Model** | Learned per-gene attention vector (`theta`, softmax-normalized as a smooth sparsemax proxy) reweights input features before a single hidden layer and classification head |
| **Loss** | Cross-entropy + `γ · Tr(Ŷᵀ L_sym Ŷ)` graph-smoothness regularization + L1 penalty on attention (encourages sparse, interpretable biomarker selection) |
| **Outputs** | Classification metrics, ranked candidate biomarkers, attention-weight distribution plot |

## Project structure

```
fgl-san/
├── fgl_san.py          # full pipeline: preprocessing → graph → model → train → eval
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
git clone https://github.com/steffid23/T-cell-biomarker-discovery-.git
cd T-cell-biomarker-discovery-
pip install -r requirements.txt
```

## Usage

```bash
# Run on your own data
python fgl_san.py --h5ad data/your_data.h5ad --label-col cell_type --epochs 350

# Run with a held-out evaluation split (recommended for reporting results)
python fgl_san.py --h5ad data/your_data.h5ad --eval-split 0.2

# No data? Runs on an auto-generated synthetic dataset for a quick smoke test
python fgl_san.py
```

## Arguments

| Flag | Default | Description |
|---|---|---|
| `--h5ad` | `data/input.h5ad` | Path to input AnnData file |
| `--label-col` | `cell_type` | `obs` column containing labels |
| `--k-neighbors` | `15` | *k* for fuzzy graph construction |
| `--epochs` | `350` | Training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--gamma` | `0.1` | Graph regularization weight |
| `--lambda-l1` | `1e-4` | L1 weight on attention (sparsity) |
| `--eval-split` | `0.0` | Held-out fraction for evaluation, e.g. `0.2`. `0.0` uses all data for training/evaluation (legacy behavior) |
| `--out-dir` | `outputs` | Where metrics/plots/weights are saved |
| `--seed` | `42` | Random seed |

## Outputs

Written to `--out-dir`:

| File | Contents |
|---|---|
| `{prefix}_Metrics.csv` | ARI, NMI, macro-F1, ROC-AUC (train + held-out if `--eval-split` used) |
| `{prefix}_Telemetry.csv` | Runtime, RAM/VRAM usage |
| `{prefix}_top_<k>_biomarkers.csv` | Genes ranked by attention score |
| `{prefix}_attention_dist.png` / `.eps` | Attention weight distribution |
| `{prefix}_weights.pt` | Trained model state dict |

## Known limitations

- **Dense Laplacian**: `L_sym` is stored as a dense `N × N` matrix, which caps practical
  scale at roughly a few thousand cells before memory becomes a bottleneck. A sparse
  Laplacian would scale further.
- **Sparsemax naming**: `sparsemax_proxy` is a temperature-scaled softmax, not true
  sparsemax/entmax — it concentrates attention but does not produce exact zeros.
- **Default evaluation**: without `--eval-split`, metrics are computed on the same data
  used for training. Use `--eval-split 0.2` (or similar) for results intended for a
  manuscript or dissertation.

## Requirements

See [`requirements.txt`](requirements.txt). Tested with Python 3.10+.
