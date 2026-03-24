"""Quick smoke test for the CEUBG pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
config.EPOCHS = 3
config.K_SHARDS = 3
config.NUM_BENCHMARK_DELETIONS = 5

from data_loader import load_data
from random_shard import RandomShard
from unlearn import UnlearningCertifier
from metrics import compute_auroc, compute_f1, compute_ddrt_accuracy, compute_kl_divergence, compute_forgetting_score, compute_mia_auc
import torch
import numpy as np

print("Loading data...")
data = load_data()
print(f"  Nodes: {data['node_features'].shape}, Train: {data['train_edges'].shape}")

print("Training Random strategy (3 shards, 3 epochs)...")
rs = RandomShard(data, k=3)
rs.train_all(epochs=3, verbose=False)
print("  Training done.")

print("Predicting test edges...")
scores_before = rs.predict(data['test_edges'])
labs = data['test_labels'].numpy()
auc = compute_auroc(labs, scores_before)
f1 = compute_f1(labs, scores_before)
print(f"  Test AUROC: {auc:.4f}, F1: {f1:.4f}")

print("Unlearning 5 edges...")
cert = UnlearningCertifier(rs)
del_edges = cert.sample_deletion_edges(data)[:5]
cert.record_model_state_before()
avg_t, tot_t, certs = cert.unlearn_sisa(del_edges)
print(f"  Avg time/del: {avg_t:.3f}s, Total: {tot_t:.3f}s")
print(f"  Certificates generated: {len(certs)}")

# Test scores after unlearning
scores_after = rs.predict(data['test_edges'])
kl = compute_kl_divergence(scores_before, scores_after)
ddrt = compute_ddrt_accuracy(
    list(rs.models.values())[0], data, del_edges, config.DEVICE
)
train_scores = rs.predict(data['train_edges'])
mia = compute_mia_auc(train_scores[:2000], scores_after[:2000])
fs = compute_forgetting_score(mia, ddrt, kl)

print(f"  KL: {kl:.4f}, DDRT: {ddrt:.4f}, MIA: {mia:.4f}, FS: {fs:.4f}")
print("\nSMOKE TEST PASSED!")
