"""
CEUBG — Data loading utilities.

Loads preprocessed .npy arrays and builds PyG-compatible data objects.
Provides balanced negative-sampling helper for per-shard training.
"""
import json
import numpy as np
import torch
from torch_geometric.data import Data

import config


def load_data():
    """
    Load all preprocessed arrays and return a dict with:
        node_features  : (N, 2048) float32 tensor
        edge_index     : (2, E)   long tensor  — global graph (pos edges only for message passing)
        edge_labels    : (E,)     float32 tensor
        train_edges    : (2, E_tr) long tensor   (includes pos + neg)
        train_labels   : (E_tr,) float32
        val_edges      : (2, E_v) long tensor
        val_labels     : (E_v,) float32
        test_edges     : (2, E_te) long tensor
        test_labels    : (E_te,) float32
        num_drugs      : int
        num_proteins   : int
    """
    node_features = torch.from_numpy(
        np.load(config.NODE_FEATURES_PATH).astype(np.float32)
    )

    # Global edge index (all edges, used for message passing)
    edge_raw = np.load(config.EDGE_INDEX_PATH)
    if edge_raw.shape[0] != 2:
        edge_raw = edge_raw.T
    edge_index = torch.from_numpy(edge_raw.astype(np.int64))

    edge_labels = torch.from_numpy(
        np.load(config.EDGE_LABELS_PATH).astype(np.float32)
    )

    # Train / val / test splits — each file stores 1D array of edge indices
    # into the main edge_index / edge_labels arrays
    train_idx = np.load(config.TRAIN_EDGES_PATH)
    val_idx   = np.load(config.VAL_EDGES_PATH)
    test_idx  = np.load(config.TEST_EDGES_PATH)

    train_edges  = edge_index[:, train_idx]
    train_labels = edge_labels[train_idx]
    val_edges    = edge_index[:, val_idx]
    val_labels   = edge_labels[val_idx]
    test_edges   = edge_index[:, test_idx]
    test_labels  = edge_labels[test_idx]


    # Build positive-only edge index for message passing
    pos_mask = edge_labels == 1.0
    pos_edge_index = edge_index[:, pos_mask] if edge_labels.shape[0] == edge_index.shape[1] else edge_index

    return {
        "node_features": node_features,
        "edge_index":    pos_edge_index,       # positive edges for GNN message passing
        "full_edge_index": edge_index,         # all edges
        "edge_labels":   edge_labels,
        "train_edges":   train_edges,
        "train_labels":  train_labels,
        "val_edges":     val_edges,
        "val_labels":    val_labels,
        "test_edges":    test_edges,
        "test_labels":   test_labels,
        "num_drugs":     config.NUM_DRUG_NODES,
        "num_proteins":  config.NUM_PROTEIN_NODES,
    }


def get_balanced_shard_edges(shard_pos_edges, num_nodes, rng=None):
    """
    Given positive edges for a shard, generate balanced negative samples.

    Args:
        shard_pos_edges: (2, E_pos) tensor of positive edges in this shard
        num_nodes: total number of nodes in the graph
        rng: optional numpy random generator
    Returns:
        edges:  (2, 2*E_pos) tensor combining pos + neg
        labels: (2*E_pos,) tensor with 1s and 0s
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pos_set = set(zip(shard_pos_edges[0].tolist(), shard_pos_edges[1].tolist()))
    num_neg = shard_pos_edges.shape[1]
    neg_src, neg_dst = [], []

    while len(neg_src) < num_neg:
        s = rng.integers(0, config.NUM_DRUG_NODES)
        d = rng.integers(config.NUM_DRUG_NODES, num_nodes)
        if (s, d) not in pos_set:
            neg_src.append(s)
            neg_dst.append(d)

    neg_edges = torch.tensor([neg_src, neg_dst], dtype=torch.long)
    edges  = torch.cat([shard_pos_edges, neg_edges], dim=1)
    labels = torch.cat([
        torch.ones(shard_pos_edges.shape[1]),
        torch.zeros(num_neg),
    ])
    return edges, labels


def load_node_mapping():
    """Load the drug/protein mapping JSON."""
    with open(config.NODE_MAPPING_PATH, "r") as f:
        return json.load(f)
