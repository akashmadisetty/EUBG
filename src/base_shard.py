"""
CEUBG — Abstract base class for all SISA sharding strategies.

All strategies inherit from this class and implement `assign_shards()`.
Training uses the GLOBAL graph for message passing; loss is restricted to shard edges.
"""
import abc
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F

import config
from model import GraphSAGEModel
from data_loader import get_balanced_shard_edges


class BaseShard(abc.ABC):
    """Abstract base class for SISA sharding strategies."""

    def __init__(self, data, k=None):
        """
        Args:
            data: dict returned by data_loader.load_data()
            k:    number of shards (default: config.K_SHARDS)
        """
        self.data = data
        self.k = k or config.K_SHARDS
        self.device = config.DEVICE
        self.models = {}          # shard_id → trained model
        self.shard_edges = {}     # shard_id → (edges, labels)  pos+neg balanced
        self.drug_to_shard = {}   # drug_node_id → shard_id
        self.shard_to_drugs = {}  # shard_id → list[drug_node_id]

    # ─── Abstract ────────────────────────────────────────────────────────
    @abc.abstractmethod
    def assign_shards(self):
        """
        Populate self.drug_to_shard and self.shard_to_drugs.
        Must partition drug nodes (0 .. NUM_DRUG_NODES-1) into self.k shards.
        """
        ...

    # ─── Build shard data ────────────────────────────────────────────────
    def build_shard_data(self):
        """
        After assign_shards(), partition train edges into shards
        and add balanced negatives for each shard.

        An edge (drug, protein) belongs to the shard owning `drug`.
        """
        train_edges  = self.data["train_edges"]  # (2, E_tr)
        train_labels = self.data["train_labels"]  # (E_tr,)
        num_nodes    = self.data["node_features"].shape[0]

        # collect positive train edges per shard
        shard_pos = {i: ([], []) for i in range(self.k)}

        for idx in range(train_edges.shape[1]):
            src = train_edges[0, idx].item()
            dst = train_edges[1, idx].item()
            lab = train_labels[idx].item()
            if src in self.drug_to_shard and lab == 1.0:
                sid = self.drug_to_shard[src]
                shard_pos[sid][0].append(src)
                shard_pos[sid][1].append(dst)

        rng = np.random.default_rng(42)
        for sid in range(self.k):
            srcs, dsts = shard_pos[sid]
            if len(srcs) == 0:
                # Empty shard — skip
                self.shard_edges[sid] = (
                    torch.zeros((2, 0), dtype=torch.long),
                    torch.zeros(0),
                )
                continue
            pos_edges = torch.tensor([srcs, dsts], dtype=torch.long)
            edges, labels = get_balanced_shard_edges(pos_edges, num_nodes, rng)
            self.shard_edges[sid] = (edges, labels)

    # ─── Train one shard ─────────────────────────────────────────────────
    def train_shard(self, shard_id, epochs=None, verbose=False):
        """
        Train a fresh GraphSAGE model for one shard.
        - Message passing: GLOBAL graph (data["edge_index"])
        - Loss: only shard edges
        """
        epochs = epochs or config.EPOCHS
        edges, labels = self.shard_edges[shard_id]
        if edges.shape[1] == 0:
            return None

        model = GraphSAGEModel(
            in_channels=config.NODE_FEATURE_DIM,
            hidden_channels=config.HIDDEN_DIM,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )

        x  = self.data["node_features"].to(self.device)
        gi = self.data["edge_index"].to(self.device)  # GLOBAL graph for message passing
        se = edges.to(self.device)
        sl = labels.to(self.device)

        best_loss = float("inf")
        patience_counter = 0

        model.train()
        for ep in range(epochs):
            optimizer.zero_grad()
            z = model.encode(x, gi)             # global message passing
            logits = model.predict(z, se)        # score shard edges only
            loss = F.binary_cross_entropy_with_logits(logits, sl)
            loss.backward()
            optimizer.step()

            l = loss.item()
            if l < best_loss - 1e-4:
                best_loss = l
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= config.PATIENCE:
                if verbose:
                    print(f"  Shard {shard_id}: early stop at epoch {ep+1}")
                break

        self.models[shard_id] = model
        return model

    # ─── Train all shards ────────────────────────────────────────────────
    def train_all(self, epochs=None, verbose=True):
        """Shard, build data, then train all shard models."""
        self.assign_shards()
        self.build_shard_data()
        for sid in range(self.k):
            if verbose:
                n_edges = self.shard_edges[sid][0].shape[1]
                print(f"  Training shard {sid}/{self.k}  ({n_edges} edges)...")
            self.train_shard(sid, epochs=epochs, verbose=verbose)

    # ─── Prediction / routing ────────────────────────────────────────────
    def predict(self, edges):
        """
        Route each edge to its owning shard model and return that model's score.
        edges: (2, E) tensor
        """
        scores = np.zeros(edges.shape[1])
        x  = self.data["node_features"].to(self.device)
        gi = self.data["edge_index"].to(self.device)

        # group edges by shard
        shard_idx_map = {}  # shard_id → list of (local_idx)
        for i in range(edges.shape[1]):
            drug = edges[0, i].item()
            sid = self.drug_to_shard.get(drug, 0)
            shard_idx_map.setdefault(sid, []).append(i)

        for sid, idxs in shard_idx_map.items():
            if sid not in self.models or self.models[sid] is None:
                continue
            model = self.models[sid]
            model.eval()
            with torch.no_grad():
                z = model.encode(x, gi)
                sub_edges = edges[:, idxs].to(self.device)
                logits = model.predict(z, sub_edges)
                s = torch.sigmoid(logits).cpu().numpy()
            for j, idx in enumerate(idxs):
                scores[idx] = s[j]

        return scores

    # ─── Remove edge and retrain ─────────────────────────────────────────
    def unlearn_edge(self, edge_src, edge_dst):
        """
        Remove a single edge from its owning shard, retrain that shard.
        Returns (shard_id, retrain_time_seconds).
        """
        sid = self.drug_to_shard.get(edge_src, 0)
        edges, labels = self.shard_edges[sid]

        # find and remove the edge
        mask = ~((edges[0] == edge_src) & (edges[1] == edge_dst))
        self.shard_edges[sid] = (edges[:, mask], labels[mask])

        t0 = time.time()
        self.train_shard(sid)
        elapsed = time.time() - t0
        return sid, elapsed

    # ─── Get embeddings ──────────────────────────────────────────────────
    def get_embeddings(self, shard_id=0):
        """Return node embeddings from a shard model (for drift analysis)."""
        model = self.models.get(shard_id)
        if model is None:
            return None
        model.eval()
        with torch.no_grad():
            x  = self.data["node_features"].to(self.device)
            gi = self.data["edge_index"].to(self.device)
            z = model.encode(x, gi)
        return z.cpu().numpy()

    @property
    def name(self):
        return self.__class__.__name__
