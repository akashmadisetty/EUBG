"""
CEUBG — End-to-end evaluation runner.

For each of 6 strategies: shard → train → evaluate → unlearn → evaluate again.
Prints an IEEE-style summary table.
"""
import sys
import os
import time
import copy
import numpy as np
import torch

# Ensure src/ is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import load_data
from metrics import (
    evaluate_model, compute_mia_auc, compute_ddrt_accuracy,
    compute_kl_divergence, compute_forgetting_score,
    compute_embedding_drift, plot_tsne,
)
from unlearn import UnlearningCertifier

from random_shard import RandomShard
from entity_shard import EntityShard
from similarity_shard import SimilarityShard
from community_shard import CommunityShard
from gradient_ascent import GradientAscentUnlearner
from full_retrain import FullRetrainUnlearner


def print_table(results):
    """Print IEEE-style summary table."""
    header = (
        f"{'Strategy':<16} {'AUROC':>7} {'F1':>7} {'MIA AUC':>8} "
        f"{'DDRT':>7} {'KL':>8} {'FS':>7} {'Avg T/Del':>10}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}")
    print("  CEUBG — Certified Edge Unlearning in Bipartite Graphs")
    print(f"  IEEE-Style Summary Table")
    print(f"{sep}")
    print(header)
    print("─" * len(header))
    for r in results:
        print(
            f"{r['strategy']:<16} {r['auroc']:>7.4f} {r['f1']:>7.4f} "
            f"{r['mia_auc']:>8.4f} {r['ddrt']:>7.4f} {r['kl']:>8.4f} "
            f"{r['fs']:>7.4f} {r['avg_time']:>9.3f}s"
        )
    print(f"{sep}\n")


def evaluate_sisa_strategy(StrategyClass, data, epochs=None, verbose=True):
    """
    Full pipeline for one SISA strategy:
    train → evaluate → unlearn → re-evaluate → compute metrics.

    Uses a deep copy of data so edge removal doesn't leak across strategies.
    """
    print(f"\n{'='*60}")
    print(f"  Strategy: {StrategyClass.__name__}")
    print(f"{'='*60}")

    # Deep copy data so edge removal in unlearn_edge doesn't affect other strategies
    local_data = {
        k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
        for k, v in data.items()
    }

    # 1. Train
    strategy = StrategyClass(local_data)
    strategy.train_all(epochs=epochs, verbose=verbose)

    # 2. Evaluate BEFORE unlearning
    test_edges  = local_data["test_edges"]
    test_labels = local_data["test_labels"]
    train_edges = local_data["train_edges"]
    train_labels = local_data["train_labels"]

    auroc, f1, test_scores = evaluate_sisa_scores(strategy, local_data, test_edges, test_labels)
    _, _, train_scores_before = evaluate_sisa_scores(strategy, local_data, train_edges, train_labels)

    # Get embeddings before
    emb_before = strategy.get_embeddings(shard_id=0)

    # Pre-unlearning scores on test edges (for KL)
    scores_before = test_scores.copy()

    print(f"  Before unlearning — AUROC: {auroc:.4f}  F1: {f1:.4f}")

    # 3. Unlearn
    certifier = UnlearningCertifier(strategy)
    deletion_edges = certifier.sample_deletion_edges(local_data)

    certifier.record_model_state_before()
    avg_time, total_time, certs = certifier.unlearn_sisa(deletion_edges)
    certifier.save_certificates()
    print(f"  Unlearned {len(deletion_edges)} edges — Avg Time/Del: {avg_time:.3f}s")

    # 4. Evaluate AFTER unlearning
    auroc_after, f1_after, test_scores_after = evaluate_sisa_scores(
        strategy, local_data, test_edges, test_labels
    )
    _, _, train_scores_after = evaluate_sisa_scores(strategy, local_data, train_edges, train_labels)

    # Get embeddings after
    emb_after = strategy.get_embeddings(shard_id=0)

    print(f"  After unlearning  — AUROC: {auroc_after:.4f}  F1: {f1_after:.4f}")

    # 5. MIA
    # Use subset of train edges (seen) and test edges (unseen)
    n_mia = min(5000, len(train_scores_after), len(test_scores_after))
    mia_auc = compute_mia_auc(
        train_scores_after[:n_mia],
        test_scores_after[:n_mia],
    )

    # 6. DDRT
    ddrt_acc = compute_ddrt_accuracy(
        strategy.models.get(0, list(strategy.models.values())[0]),
        local_data, deletion_edges, config.DEVICE,
    )

    # 7. KL
    kl = compute_kl_divergence(scores_before, test_scores_after)

    # 8. Forgetting Score
    fs = compute_forgetting_score(mia_auc, ddrt_acc, kl)

    # 9. Embedding drift + t-SNE
    if emb_before is not None and emb_after is not None:
        drift = compute_embedding_drift(emb_before, emb_after)
        try:
            plot_tsne(emb_before, emb_after, strategy.name)
        except Exception:
            pass
        print(f"  Embedding L2 drift: {drift:.4f}")

    print(f"  MIA AUC: {mia_auc:.4f}  DDRT: {ddrt_acc:.4f}  KL: {kl:.4f}  FS: {fs:.4f}")

    return {
        "strategy": strategy.name,
        "auroc": auroc_after,
        "f1": f1_after,
        "mia_auc": mia_auc,
        "ddrt": ddrt_acc,
        "kl": kl,
        "fs": fs,
        "avg_time": avg_time,
    }


def evaluate_sisa_scores(strategy, data, edges, labels):
    """Get scores from SISA strategy (routes per edge)."""
    scores = strategy.predict(edges)
    labs = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    from metrics import compute_auroc, compute_f1
    auroc = compute_auroc(labs, scores)
    f1    = compute_f1(labs, scores)
    return auroc, f1, scores


def evaluate_global_unlearner(UnlearnerClass, data, label, epochs=None, verbose=True):
    """
    Full pipeline for global (non-shard) unlearning strategies:
    GradientAscent and FullRetrain.

    Uses a deep copy of data so edge removal doesn't leak across strategies.
    """
    print(f"\n{'='*60}")
    print(f"  Strategy: {label}")
    print(f"{'='*60}")

    # Deep copy data so edge removal doesn't affect other strategies
    local_data = {
        k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
        for k, v in data.items()
    }

    unlearner = UnlearnerClass(local_data)
    unlearner.train_global(epochs=epochs, verbose=verbose)

    test_edges  = local_data["test_edges"]
    test_labels = local_data["test_labels"]
    train_edges = local_data["train_edges"]

    # Evaluate before
    scores_before = unlearner.predict(test_edges)
    train_scores = unlearner.predict(train_edges)
    labs = test_labels.numpy()
    from metrics import compute_auroc, compute_f1
    auroc = compute_auroc(labs, scores_before)
    f1    = compute_f1(labs, scores_before)
    print(f"  Before unlearning — AUROC: {auroc:.4f}  F1: {f1:.4f}")

    emb_before = unlearner.get_embeddings()

    # Unlearn
    certifier = UnlearningCertifier(unlearner)
    deletion_edges = certifier.sample_deletion_edges(local_data)

    certifier.record_model_state_before()

    if isinstance(unlearner, GradientAscentUnlearner):
        # Convert to tensor for GA batch unlearning
        srcs = [e[0] for e in deletion_edges]
        dsts = [e[1] for e in deletion_edges]
        forget_tensor = torch.tensor([srcs, dsts], dtype=torch.long)
        avg_time, total_time, certs = certifier.unlearn_gradient_ascent(deletion_edges, local_data)
    elif isinstance(unlearner, FullRetrainUnlearner):
        avg_time, total_time, certs = certifier.unlearn_full_retrain(deletion_edges, local_data)
    else:
        raise ValueError(f"Unknown unlearner type: {type(unlearner)}")

    certifier.save_certificates()
    print(f"  Unlearned {len(deletion_edges)} edges — Avg Time/Del: {avg_time:.4f}s  Total: {total_time:.2f}s")

    # Evaluate after
    scores_after = unlearner.predict(test_edges)
    train_scores_after = unlearner.predict(train_edges)
    auroc_after = compute_auroc(labs, scores_after)
    f1_after    = compute_f1(labs, scores_after)
    print(f"  After unlearning  — AUROC: {auroc_after:.4f}  F1: {f1_after:.4f}")

    emb_after = unlearner.get_embeddings()

    # MIA
    n_mia = min(5000, len(train_scores_after), len(scores_after))
    mia_auc = compute_mia_auc(train_scores_after[:n_mia], scores_after[:n_mia])

    # DDRT
    ddrt_acc = compute_ddrt_accuracy(unlearner.model, local_data, deletion_edges, config.DEVICE)

    # KL
    kl = compute_kl_divergence(scores_before, scores_after)

    # FS
    fs = compute_forgetting_score(mia_auc, ddrt_acc, kl)

    # Embedding drift + t-SNE
    if emb_before is not None and emb_after is not None:
        drift = compute_embedding_drift(emb_before, emb_after)
        try:
            plot_tsne(emb_before, emb_after, label)
        except Exception:
            pass
        print(f"  Embedding L2 drift: {drift:.4f}")

    print(f"  MIA AUC: {mia_auc:.4f}  DDRT: {ddrt_acc:.4f}  KL: {kl:.4f}  FS: {fs:.4f}")

    return {
        "strategy": label,
        "auroc": auroc_after,
        "f1": f1_after,
        "mia_auc": mia_auc,
        "ddrt": ddrt_acc,
        "kl": kl,
        "fs": fs,
        "avg_time": avg_time,
    }


def main():
    print("Loading data...")
    data = load_data()
    print(f"  Nodes: {data['node_features'].shape[0]}  Features: {data['node_features'].shape[1]}")
    print(f"  Train edges: {data['train_edges'].shape[1]}  Val: {data['val_edges'].shape[1]}  Test: {data['test_edges'].shape[1]}")

    # Optionally reduce epochs for quick smoke test
    epochs = config.EPOCHS

    results = []

    # 1. Random
    results.append(evaluate_sisa_strategy(RandomShard, data, epochs=epochs))

    # 2. Entity
    results.append(evaluate_sisa_strategy(EntityShard, data, epochs=epochs))

    # 3. Similarity
    results.append(evaluate_sisa_strategy(SimilarityShard, data, epochs=epochs))

    # 4. Community
    results.append(evaluate_sisa_strategy(CommunityShard, data, epochs=epochs))

    # 5. Gradient Ascent
    results.append(evaluate_global_unlearner(GradientAscentUnlearner, data, "GradientAscent", epochs=epochs))

    # 6. Full Retrain (gold standard baseline)
    results.append(evaluate_global_unlearner(FullRetrainUnlearner, data, "FullRetrain", epochs=epochs))

    # ─── Summary table ───────────────────────────────────────────────────
    print_table(results)

    # Save results as JSON
    import json
    results_path = os.path.join(config.OUTPUT_DIR, "summary_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
