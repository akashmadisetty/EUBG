"""
CEUBG — Gradient Ascent Unlearning (non-SISA alternative).

No sharding. On the global trained model, flip BCE sign for gradient
steps on the forgotten edges to "unlearn" them. Uses gradient clipping
and utility-preservation early stopping to avoid catastrophic model collapse.
"""
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F

import config
from model import GraphSAGEModel
from data_loader import get_balanced_shard_edges


class GradientAscentUnlearner:
    """
    Gradient-ascent unlearning on the global model.
    No sharding — trains one global model, then unlearns via gradient ascent.
    """

    def __init__(self, data):
        self.data = data
        self.device = config.DEVICE
        self.model = None

    @property
    def name(self):
        return "GradientAscent"

    # ─── Train global model ──────────────────────────────────────────────
    def train_global(self, epochs=None, verbose=True):
        """Train a single global GraphSAGE model on ALL training edges."""
        epochs = epochs or config.EPOCHS

        # Fixed seed for reproducible model initialization (fair comparison with FullRetrain)
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

    # ─── Gradient ascent unlearning ──────────────────────────────────────
    def unlearn_edges(self, edges_to_forget, steps=None, lr=None):
        """
        Flip BCE sign and do gradient ASCENT on the forgotten edges.
        Removes forgotten edges from the message-passing graph to prevent
        data leakage. Uses gradient clipping and utility-preservation
        early stopping to avoid catastrophic model collapse.

        Args:
            edges_to_forget: (2, E_forget) tensor of edges to unlearn
            steps: number of GA steps (default: config.GA_STEPS)
            lr:    learning rate for GA (default: config.GA_LR)
        Returns:
            total_time: time in seconds for the entire GA process
        """
        steps = steps or config.GA_STEPS
        lr = lr or config.GA_LR

        if self.model is None:
            raise RuntimeError("Must call train_global() first")

        # ── Remove forgotten edges from message-passing graph (data leakage fix) ──
        gi = self.data["edge_index"]
        for i in range(edges_to_forget.shape[1]):
            src, dst = edges_to_forget[0, i].item(), edges_to_forget[1, i].item()
            mask = ~((gi[0] == src) & (gi[1] == dst))
            mask &= ~((gi[0] == dst) & (gi[1] == src))  # reverse direction
            gi = gi[:, mask]
        self.data["edge_index"] = gi

        x  = self.data["node_features"].to(self.device)
        gi = self.data["edge_index"].to(self.device)
        forget_edges = edges_to_forget.to(self.device)

        # Forgotten edges were positive — labels = 1
        forget_labels = torch.ones(forget_edges.shape[1], device=self.device)

        # Sample retained train edges for utility-preservation check
        train_edges = self.data["train_edges"].to(self.device)
        train_labels = self.data["train_labels"].to(self.device)
        n_util = min(2000, train_edges.shape[1])
        util_idx = torch.randperm(train_edges.shape[1])[:n_util]
        util_edges = train_edges[:, util_idx]
        util_labels = train_labels[util_idx]

        # Measure baseline utility loss before GA
        self.model.eval()
        with torch.no_grad():
            z0 = self.model.encode(x, gi)
            baseline_logits = self.model.predict(z0, util_edges)
            baseline_util_loss = F.binary_cross_entropy_with_logits(
                baseline_logits, util_labels
            ).item()

        # Save model checkpoint in case of collapse
        checkpoint = copy.deepcopy(self.model.state_dict())

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        t0 = time.time()
        best_state = None
        for step in range(steps):
            optimizer.zero_grad()
            z = self.model.encode(x, gi)
            logits = self.model.predict(z, forget_edges)
            # FLIP THE SIGN: maximize loss = train model to FORGET these edges
            loss = -F.binary_cross_entropy_with_logits(logits, forget_labels)
            loss.backward()

            # Gradient clipping to prevent catastrophic parameter drift
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), config.GA_CLIP_NORM
            )
            optimizer.step()

            # Utility-preservation check every 5 steps
            if (step + 1) % 5 == 0 or step == steps - 1:
                self.model.eval()
                with torch.no_grad():
                    z_check = self.model.encode(x, gi)
                    util_logits = self.model.predict(z_check, util_edges)
                    util_loss = F.binary_cross_entropy_with_logits(
                        util_logits, util_labels
                    ).item()

                # If utility degrades too much (>3x baseline), stop early
                if util_loss > max(baseline_util_loss * 3.0, 1.5):
                    if best_state is not None:
                        self.model.load_state_dict(best_state)
                    break

                # Track best model state (lowest utility degradation)
                if best_state is None or util_loss <= baseline_util_loss * 1.5:
                    best_state = copy.deepcopy(self.model.state_dict())
                self.model.train()

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
