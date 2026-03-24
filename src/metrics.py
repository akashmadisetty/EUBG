"""
CEUBG — Unified metrics suite.

AUROC, F1, MIA AUC, DDRT accuracy, KL divergence, Forgetting Score,
embedding L2 drift, and t-SNE plotting.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os, json

import config


# ═══════════════════════════════════════════════════════════════════════════
#  Standard link prediction metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_auroc(labels, scores):
    """AUROC from ground-truth labels and predicted scores."""
    return roc_auc_score(labels, scores)


def compute_f1(labels, scores, threshold=0.5):
    """F1 score with decision threshold."""
    preds = (np.array(scores) >= threshold).astype(int)
    return f1_score(labels, preds)


# ═══════════════════════════════════════════════════════════════════════════
#  Membership Inference Attack (MIA) AUC
# ═══════════════════════════════════════════════════════════════════════════

def compute_mia_auc(member_scores, non_member_scores):
    """
    Train a logistic regression adversary:
        - member_scores:     model's prediction scores on train edges (seen)
        - non_member_scores: model's prediction scores on test edges (unseen)
    Lower AUC → better privacy (adversary can't distinguish).
    """
    X = np.concatenate([
        np.array(member_scores).reshape(-1, 1),
        np.array(non_member_scores).reshape(-1, 1),
    ])
    y = np.concatenate([
        np.ones(len(member_scores)),
        np.zeros(len(non_member_scores)),
    ])
    # Shuffle
    idx = np.random.RandomState(42).permutation(len(y))
    X, y = X[idx], y[idx]

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    return roc_auc_score(y, probs)


# ═══════════════════════════════════════════════════════════════════════════
#  Deleted Data Retention Test (DDRT)
# ═══════════════════════════════════════════════════════════════════════════

def compute_ddrt_accuracy(model, data, deleted_edges, device):
    """
    Query the retrained model on deleted edges.
    Returns accuracy — lower accuracy = better forgetting.

    The deleted edges should all have been positive edges,
    so if the model scores them high (predicts them as positive),
    forgetting is poor.
    """
    model.eval()
    with torch.no_grad():
        x = data["node_features"].to(device)
        ei = data["edge_index"].to(device)
        z = model.encode(x, ei)

        del_edges = torch.tensor(deleted_edges, dtype=torch.long).to(device)
        if del_edges.dim() == 2 and del_edges.shape[0] != 2:
            del_edges = del_edges.T

        scores = torch.sigmoid(model.predict(z, del_edges)).cpu().numpy()

    # These were all positive edges, so "remembered" = score > 0.5
    preds = (scores >= 0.5).astype(int)
    # Accuracy of predicting them as positive — lower = better forgetting
    accuracy = np.mean(preds)
    return accuracy


# ═══════════════════════════════════════════════════════════════════════════
#  KL Divergence
# ═══════════════════════════════════════════════════════════════════════════

def compute_kl_divergence(scores_before, scores_after, eps=1e-8):
    """
    KL(before || after) on the prediction-score distributions.
    Uses histogram-based estimation.
    """
    bins = np.linspace(0, 1, 51)
    p, _ = np.histogram(scores_before, bins=bins, density=True)
    q, _ = np.histogram(scores_after,  bins=bins, density=True)
    p = p.astype(np.float64) + eps
    q = q.astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


# ═══════════════════════════════════════════════════════════════════════════
#  Forgetting Score
# ═══════════════════════════════════════════════════════════════════════════

def compute_forgetting_score(mia_auc, ddrt_acc, kl_div):
    """
    FS = 1/3 * (1 - |MIA - 0.5| / 0.5)
       + 1/3 * (1 - DDRT_acc)
       + 1/3 * exp(-KL)

    NOTE: uses exp(-KL), NOT 1 - exp(-KL)
    """
    mia_term  = 1.0 - abs(mia_auc - 0.5) / 0.5
    ddrt_term = 1.0 - ddrt_acc
    kl_term   = np.exp(-kl_div)
    return (mia_term + ddrt_term + kl_term) / 3.0


# ═══════════════════════════════════════════════════════════════════════════
#  Embedding drift
# ═══════════════════════════════════════════════════════════════════════════

def compute_embedding_drift(emb_before, emb_after):
    """Average L2 norm of per-node embedding change."""
    diff = emb_after - emb_before
    norms = np.linalg.norm(diff, axis=1)
    return float(np.mean(norms))


# ═══════════════════════════════════════════════════════════════════════════
#  t-SNE Visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_tsne(emb_before, emb_after, strategy_name, save_dir=None):
    """
    2D t-SNE of node embeddings before/after unlearning.
    Saves to config.PLOTS_DIR.
    """
    if save_dir is None:
        save_dir = config.PLOTS_DIR

    n = min(5000, emb_before.shape[0])  # subsample for speed
    idx = np.random.RandomState(42).choice(emb_before.shape[0], n, replace=False)

    combined = np.vstack([emb_before[idx], emb_after[idx]])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(combined)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(coords[:n, 0], coords[:n, 1], s=3, alpha=0.4, label="Before")
    ax.scatter(coords[n:, 0], coords[n:, 1], s=3, alpha=0.4, label="After")
    ax.set_title(f"t-SNE Embedding Drift — {strategy_name}")
    ax.legend()
    path = os.path.join(save_dir, f"tsne_{strategy_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Full evaluation helper
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, data, edges, labels, device):
    """
    Run model on given edges and return (auroc, f1, raw_scores).
    """
    model.eval()
    with torch.no_grad():
        x  = data["node_features"].to(device)
        ei = data["edge_index"].to(device)
        z  = model.encode(x, ei)

        e = edges.to(device)
        logits = model.predict(z, e)
        scores = torch.sigmoid(logits).cpu().numpy()

    labs = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    auroc = compute_auroc(labs, scores)
    f1    = compute_f1(labs, scores)
    return auroc, f1, scores
