"""
CEUBG — Full Retraining Baseline.

Gold-standard unlearning: removes deleted edges from training data AND
message-passing graph, then retrains the model from scratch.
This is the ideal outcome any approximate unlearning method should aim for.
"""
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F

import config
from model import GraphSAGEModel


class FullRetrainUnlearner:
    """
    Full retraining unlearning baseline.
    Trains a global model, then for unlearning: removes deleted edges
    from both training data and message-passing graph, retrains from scratch.
    """

    def __init__(self, data):
        self.data = data
        self.device = config.DEVICE
        self.model = None

    @property
    def name(self):
        return "FullRetrain"

    # ─── Train global model ──────────────────────────────────────────────
    def train_global(self, epochs=None, verbose=True):
        """Train a single global GraphSAGE model on ALL training edges."""
        epochs = epochs or config.EPOCHS

        # Fixed seed for reproducible model initialization (fair comparison with GradientAscent)
        torch.manual_seed(42)
        self.model = GraphSAGEModel(
            in_channels=config.NODE_FEATURE_DIM,
            hidden_channels=config.HIDDEN_DIM,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        x  = self.data["node_features"].to(self.device)
        gi = self.data["edge_index"].to(self.device)
        te = self.data["train_edges"].to(self.device)
        tl = self.data["train_labels"].to(self.device)

        best_loss = float("inf")
        patience_counter = 0

        self.model.train()
        for ep in range(epochs):
            optimizer.zero_grad()
            z = self.model.encode(x, gi)
            logits = self.model.predict(z, te)
            loss = F.binary_cross_entropy_with_logits(logits, tl)
            loss.backward()
            optimizer.step()

            l = loss.item()
            if verbose and (ep + 1) % 50 == 0:
                print(f"    Global epoch {ep+1}/{epochs}  loss={l:.4f}")
            if l < best_loss - 1e-4:
                best_loss = l
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= config.PATIENCE:
                if verbose:
                    print(f"    Early stop at epoch {ep+1}")
                break

    # ─── Full retrain unlearning ─────────────────────────────────────────
    def unlearn_edges(self, deletion_edges, epochs=None, verbose=False):
        """
        Remove deleted edges from training data and message-passing graph,
        then retrain the model from scratch.

        Args:
            deletion_edges: list of (src, dst) tuples to forget
            epochs: training epochs for retraining (default: config.FULL_RETRAIN_EPOCHS)
            verbose: print training progress
        Returns:
            total_time: time in seconds for the entire retrain process
        """
        epochs = epochs or config.FULL_RETRAIN_EPOCHS

        # ── Remove deleted edges from training data ──
        train_edges = self.data["train_edges"]
        train_labels = self.data["train_labels"]

        mask = torch.ones(train_edges.shape[1], dtype=torch.bool)
        for src, dst in deletion_edges:
            edge_mask = (train_edges[0] == src) & (train_edges[1] == dst)
            mask &= ~edge_mask
        self.data["train_edges"] = train_edges[:, mask]
        self.data["train_labels"] = train_labels[mask]

        # ── Remove from message-passing graph (data leakage fix) ──
        gi = self.data["edge_index"]
        for src, dst in deletion_edges:
            gi_mask = ~((gi[0] == src) & (gi[1] == dst))
            gi_mask &= ~((gi[0] == dst) & (gi[1] == src))  # reverse direction
            gi = gi[:, gi_mask]
        self.data["edge_index"] = gi

        # ── Retrain from scratch ──
        t0 = time.time()
        torch.manual_seed(42)
        self.model = GraphSAGEModel(
            in_channels=config.NODE_FEATURE_DIM,
            hidden_channels=config.HIDDEN_DIM,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        x  = self.data["node_features"].to(self.device)
        gi = self.data["edge_index"].to(self.device)
        te = self.data["train_edges"].to(self.device)
        tl = self.data["train_labels"].to(self.device)

        best_loss = float("inf")
        patience_counter = 0

        self.model.train()
        for ep in range(epochs):
            optimizer.zero_grad()
            z = self.model.encode(x, gi)
            logits = self.model.predict(z, te)
            loss = F.binary_cross_entropy_with_logits(logits, tl)
            loss.backward()
            optimizer.step()

            l = loss.item()
            if verbose and (ep + 1) % 50 == 0:
                print(f"    Retrain epoch {ep+1}/{epochs}  loss={l:.4f}")
            if l < best_loss - 1e-4:
                best_loss = l
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= config.PATIENCE:
                if verbose:
                    print(f"    Early stop at epoch {ep+1}")
                break

        total_time = time.time() - t0
        return total_time

    # ─── Predict ─────────────────────────────────────────────────────────
    def predict(self, edges):
        """Score edges using the global model."""
        self.model.eval()
        with torch.no_grad():
            x  = self.data["node_features"].to(self.device)
            gi = self.data["edge_index"].to(self.device)
            z = self.model.encode(x, gi)
            e = edges.to(self.device)
            logits = self.model.predict(z, e)
            scores = torch.sigmoid(logits).cpu().numpy()
        return scores

    # ─── Embeddings ──────────────────────────────────────────────────────
    def get_embeddings(self):
        """Return node embeddings from the global model."""
        self.model.eval()
        with torch.no_grad():
            x  = self.data["node_features"].to(self.device)
            gi = self.data["edge_index"].to(self.device)
            z = self.model.encode(x, gi)
        return z.cpu().numpy()
