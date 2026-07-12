# FGL-SAN: Fuzzy Graph Laplacian-Regularized Shallow Attention Network

A shallow attention network for single-cell / spatial transcriptomics data that combines a
sparse-attention feature-weighting layer with graph-Laplacian regularization derived from a
fuzzy k-nearest-neighbor similarity graph.

## Method overview

1. **Preprocessing** — filter cells/genes, CPM-normalize, log1p, scale (`scanpy`).
2. **Fuzzy graph construction** — build a kNN graph with adaptive (self-tuning) Gaussian
   bandwidths per cell, symmetrize, and compute the normalized graph Laplacian `L_sym`.
3. **Model** — a shallow network with a learned per-gene attention vector (`theta`,
   softmax-normalized as a smooth sparsemax proxy) that reweights input features before a
   single hidden layer and classification head.
4. **Loss** — cross-entropy + `gamma * Tr(Y_hat^T L_sym Y_hat)` (graph smoothness
   regularization) + L1 penalty on attention weights (encourages sparse biomarker selection).
5. **Outputs** — classification metrics (ARI, NMI, macro-F1, ROC-AUC), top attended genes
   (candidate biomarkers) as CSV, and an attention-weight distribution plot.

## Usage

```bash
pip install -r requirements.txt
python fgl_san.py --h5ad data/your_data.h5ad --label-col cell_type --epochs 350
```

Runs on an auto-generated synthetic dataset if no `--h5ad` path is given, for quick testing.

### Key arguments

| Flag | Default | Description |
|---|---|---|
| `--h5ad` | `data/input.h5ad` | Path to input AnnData file |
| `--label-col` | `cell_type` | `obs` column containing labels |
| `--k-neighbors` | 15 | k for fuzzy graph construction |
| `--epochs` | 350 | Training epochs |
| `--gamma` | 0.1 | Graph regularization weight |
| `--lambda-l1` | 1e-4 | L1 weight on attention (sparsity) |
| `--out-dir` | `outputs` | Where metrics/plots/weights are saved |
| `--seed` | 42 | Random seed |

## Outputs (written to `--out-dir`)

- `{prefix}_Metrics.csv` — ARI, NMI, F1, ROC-AUC
- `{prefix}_Telemetry.csv` — runtime, RAM/VRAM usage
- `{prefix}_top_<k>_biomarkers.csv` — genes ranked by attention score
- `{prefix}_attention_dist.png/.eps` — attention weight distribution
- `{prefix}_weights.pt` — trained model state dict

## Known limitations

- Metrics are currently computed on the full training set (no held-out split). For
  performance claims in a manuscript, add a train/test or train/val split.
- The graph Laplacian is stored densely (`N x N`), which limits scalability to roughly a
  few thousand cells before memory becomes a bottleneck; a sparse Laplacian would scale further.
- `sparsemax_proxy` is a temperature-scaled softmax, not true sparsemax/entmax — it does not
  produce exact sparsity, only soft concentration.

## Requirements

See `requirements.txt`. Tested with Python 3.10+.
