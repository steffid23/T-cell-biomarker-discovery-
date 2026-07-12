# -*- coding: utf-8 -*-
"""
Fuzzy Graph Laplacian-Regularized Shallow Attention Networks (FGL-SAN)

Usage:
    python fgl_san.py --h5ad path/to/data.h5ad --label-col cell_type --epochs 350
    python fgl_san.py                      # runs on synthetic demo data
"""

import os
import time
import argparse
import random

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score, f1_score
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_telemetry(task_name, start_time):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ram_mb = mem_info.rss / (1024 ** 2)

    vram_mb = 0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    duration = time.time() - start_time

    return {
        "Task": task_name,
        "Duration_s": round(duration, 2),
        "RAM_MB": round(ram_mb, 2),
        "VRAM_MB": round(vram_mb, 2)
    }


def preprocess_and_build_graph(h5ad_path, label_col='cell_type', k_neighbors=15):
    print(f"Loading data from {h5ad_path}...")
    start_time = time.time()

    if not os.path.exists(h5ad_path):
        print(f"File not found: {h5ad_path}. Generating SYNTHETIC dataset for demonstration...")
        n_cells, n_genes = 2000, 500
        X = np.random.poisson(lam=0.5, size=(n_cells, n_genes)).astype(np.float32)
        labels = np.random.choice([0, 1, 2], size=n_cells)
        X[labels == 0, :50] += 2
        X[labels == 1, 50:100] += 2
        X[labels == 2, 100:150] += 2

        adata = sc.AnnData(X=csr_matrix(X))
        adata.obs[label_col] = labels.astype(str)
    else:
        adata = sc.read_h5ad(h5ad_path)

    print("Filtering cells and genes...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    print("Normalizing (CPM) and scaling...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)

    unique_labels = adata.obs[label_col].unique()
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in adata.obs[label_col]])

    print("Building Fuzzy Graph Laplacian...")
    X_dense = adata.X.toarray() if isinstance(adata.X, csr_matrix) else adata.X
    tree = cKDTree(X_dense)
    distances, indices = tree.query(X_dense, k=k_neighbors + 1)

    sigma = distances[:, -1]
    sigma = np.maximum(sigma, 1e-5)

    N = X_dense.shape[0]

    # Vectorized similarity construction (replaces the O(N^2) Python double loop).
    rows = np.repeat(np.arange(N), k_neighbors)
    cols = indices[:, 1:].flatten()
    dist_sq = (distances[:, 1:].flatten()) ** 2
    sigma_i = np.repeat(sigma, k_neighbors)
    sigma_j = sigma[cols]
    vals = np.exp(-dist_sq / (2 * sigma_i * sigma_j))

    S = np.zeros((N, N), dtype=np.float32)
    S[rows, cols] = vals

    # Symmetrize
    S = np.maximum(S, S.T)

    d = np.sum(S, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(N) - D_inv_sqrt @ S @ D_inv_sqrt

    tel = log_telemetry("Preprocessing & Graph Build", start_time)
    print(f"Preprocessing finished. {tel}")

    return torch.FloatTensor(X_dense), torch.FloatTensor(L_sym), torch.LongTensor(y), adata.var_names.values, label_map


def sparsemax_proxy(x, tau=0.1):
    """Temperature-scaled softmax used as a smooth, differentiable proxy for sparsemax/entmax."""
    return F.softmax(x / tau, dim=-1)


class GraphRegShallowAttention(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(n_features))
        self.W1 = nn.Parameter(torch.randn(n_features, 64) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(64))
        self.W2 = nn.Parameter(torch.randn(64, n_classes) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(n_classes))

    def forward(self, X):
        alpha = sparsemax_proxy(self.theta)
        X_tilde = X * alpha
        H = F.leaky_relu(X_tilde @ self.W1 + self.b1)
        Y_hat_logits = H @ self.W2 + self.b2
        Y_hat = F.softmax(Y_hat_logits, dim=-1)
        return Y_hat, alpha


def train_model(X, L_sym, y, train_idx=None, n_epochs=150, lr=1e-3, gamma=0.1, lambda_l1=1e-4):
    X = X.to(device)
    L_sym = L_sym.to(device)
    y = y.to(device)

    if train_idx is not None:
        train_idx = torch.LongTensor(train_idx).to(device)

    n_features = X.shape[1]
    n_classes = len(torch.unique(y))

    model = GraphRegShallowAttention(n_features, n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    start_time = time.time()

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        Y_hat, alpha = model(X)

        # Graph regularization uses the full graph (transductive), but the
        # supervised loss only sees training-set labels when a split is given.
        if train_idx is not None:
            loss_ce = F.cross_entropy(Y_hat[train_idx], y[train_idx])
        else:
            loss_ce = F.cross_entropy(Y_hat, y)
        loss_reg = gamma * torch.trace(Y_hat.T @ L_sym @ Y_hat)
        loss_l1 = lambda_l1 * torch.sum(torch.abs(model.theta))

        loss = loss_ce + loss_reg + loss_l1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f} | CE: {loss_ce.item():.4f} | Reg: {loss_reg.item():.4f}")

    tel = log_telemetry("Model Training", start_time)
    print(f"Training finished. {tel}")
    return model, tel


def _compute_metrics(y_true, Y_pred, Y_prob):
    ari = adjusted_rand_score(y_true, Y_pred)
    nmi = normalized_mutual_info_score(y_true, Y_pred)
    f1 = f1_score(y_true, Y_pred, average='macro')
    try:
        roc = roc_auc_score(y_true, Y_prob, multi_class='ovr')
    except Exception:
        roc = np.nan
    return ari, nmi, f1, roc


def evaluate_and_plot(model, X, y, var_names, out_dir, prefix="full_model",
                       train_idx=None, test_idx=None):
    model.eval()
    with torch.no_grad():
        Y_hat, alpha = model(X.to(device))
        Y_pred = torch.argmax(Y_hat, dim=1).cpu().numpy()
        Y_prob = Y_hat.cpu().numpy()
        y_true = y.cpu().numpy()

    metrics = {"Model": prefix}

    if test_idx is not None:
        ari, nmi, f1, roc = _compute_metrics(y_true[test_idx], Y_pred[test_idx], Y_prob[test_idx])
        metrics.update({"Test_ARI": ari, "Test_NMI": nmi, "Test_F1_Macro": f1, "Test_ROC_AUC": roc})
        print(f"[Held-out test] {prefix}: ARI={ari:.4f}, NMI={nmi:.4f}, F1={f1:.4f}, ROC={roc:.4f}")

        ari_tr, nmi_tr, f1_tr, roc_tr = _compute_metrics(y_true[train_idx], Y_pred[train_idx], Y_prob[train_idx])
        metrics.update({"Train_ARI": ari_tr, "Train_NMI": nmi_tr, "Train_F1_Macro": f1_tr, "Train_ROC_AUC": roc_tr})
        print(f"[Train] {prefix}: ARI={ari_tr:.4f}, NMI={nmi_tr:.4f}, F1={f1_tr:.4f}, ROC={roc_tr:.4f}")
    else:
        ari, nmi, f1, roc = _compute_metrics(y_true, Y_pred, Y_prob)
        metrics.update({"ARI": ari, "NMI": nmi, "F1_Macro": f1, "ROC_AUC": roc})
        print(f"Metrics for {prefix} (no held-out split): ARI={ari:.4f}, NMI={nmi:.4f}, F1={f1:.4f}, ROC={roc:.4f}")

    alpha_np = alpha.cpu().numpy()
    top_k = min(1500, len(alpha_np))
    top_indices = np.argsort(alpha_np)[::-1][:top_k]
    top_genes = var_names[top_indices]
    top_scores = alpha_np[top_indices]

    df_bio = pd.DataFrame({
        'Biomarker_Name': top_genes,
        'Original_Index': top_indices,
        'Attention_Score': top_scores
    })
    csv_path = os.path.join(out_dir, f"{prefix}_top_{top_k}_biomarkers.csv")
    df_bio.to_csv(csv_path, index=False)
    print(f"Saved top {top_k} biomarkers to {csv_path}")

    plt.figure(figsize=(10, 5))
    sns.histplot(alpha_np, bins=100, kde=True)
    plt.title(f"Attention Weights Distribution ({prefix})")
    plt.xlabel("Attention Score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(out_dir, f"{prefix}_attention_dist.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, f"{prefix}_attention_dist.eps"), format='eps', bbox_inches='tight')
    plt.close()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="FGL-SAN: Fuzzy Graph Laplacian-Regularized Shallow Attention Network")
    parser.add_argument("--h5ad", type=str, default="data/input.h5ad", help="Path to input .h5ad file")
    parser.add_argument("--label-col", type=str, default="cell_type", help="obs column with cell type labels")
    parser.add_argument("--k-neighbors", type=int, default=15, help="k for the fuzzy graph")
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.1, help="Graph regularization weight")
    parser.add_argument("--lambda-l1", type=float, default=1e-4, help="L1 weight on attention")
    parser.add_argument("--prefix", type=str, default="Model_Full")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-split", type=float, default=0.0,
                         help="Held-out fraction for honest evaluation (e.g. 0.2). "
                              "0.0 keeps the original full-data behavior.")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Using device: {device}")

    X, L_sym, y, var_names, label_map = preprocess_and_build_graph(
        args.h5ad, label_col=args.label_col, k_neighbors=args.k_neighbors
    )

    train_idx, test_idx = None, None
    if args.eval_split > 0:
        from sklearn.model_selection import train_test_split
        all_idx = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            all_idx, test_size=args.eval_split, stratify=y.numpy(), random_state=args.seed
        )

    model, tel = train_model(
        X, L_sym, y, train_idx=train_idx,
        n_epochs=args.epochs, lr=args.lr, gamma=args.gamma, lambda_l1=args.lambda_l1
    )

    metrics = evaluate_and_plot(
        model, X, y, var_names, args.out_dir, prefix=args.prefix,
        train_idx=train_idx, test_idx=test_idx
    )

    pd.DataFrame([metrics]).to_csv(os.path.join(args.out_dir, f"{args.prefix}_Metrics.csv"), index=False)
    pd.DataFrame([tel]).to_csv(os.path.join(args.out_dir, f"{args.prefix}_Telemetry.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"{args.prefix}_weights.pt"))

    print("Pipeline complete. Artifacts saved to", args.out_dir)


if __name__ == "__main__":
    main()
