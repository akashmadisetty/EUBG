"""
CEUBG — Unlearning procedure with SHA-256 cryptographic certificates.

Sample 100 benchmark deletion edges (seed=123).
For each: hash model weights before → remove edge → retrain shard →
hash after → emit JSON certificate.
Reports Avg Time / Deletion (not total batch time).
"""
import hashlib
import json
import time
import os
import copy
import numpy as np
import torch

import config


class UnlearningCertifier:
    """
    Handles the certified unlearning workflow for any strategy.
    Works with both SISA-based sharding strategies (BaseShard subclasses)
    and the GradientAscent unlearner.
    """

    def __init__(self, strategy):
        """
        Args:
            strategy: one of the shard classes or GradientAscentUnlearner
        """
        self.strategy = strategy
        self.certificates = []

    # ─── SHA-256 of model weights ────────────────────────────────────────
    @staticmethod
    def hash_model(model):
        """SHA-256 hash of all model parameters (flattened bytes)."""
        h = hashlib.sha256()
        for param in model.parameters():
            h.update(param.data.cpu().numpy().tobytes())
        return h.hexdigest()

    # ─── Record model state before unlearning ────────────────────────────
    def record_model_state_before(self):
        """Snapshot hashes of all shard models (or global model)."""
        self._pre_hashes = {}
        if hasattr(self.strategy, "models"):
            # SISA strategies
            for sid, model in self.strategy.models.items():
                if model is not None:
                    self._pre_hashes[sid] = self.hash_model(model)
        elif hasattr(self.strategy, "model") and self.strategy.model is not None:
            # Gradient Ascent
            self._pre_hashes["global"] = self.hash_model(self.strategy.model)

    # ─── Sample benchmark edges ──────────────────────────────────────────
    def sample_deletion_edges(self, data):
        """
        Sample NUM_BENCHMARK_DELETIONS positive train edges for unlearning.
        """
        rng = np.random.default_rng(config.UNLEARN_SEED)
        train_edges  = data["train_edges"]
        train_labels = data["train_labels"]

        # collect positive edges
        pos_mask = train_labels == 1.0
        pos_indices = torch.where(pos_mask)[0].numpy()

        chosen = rng.choice(
            pos_indices,
            size=min(config.NUM_BENCHMARK_DELETIONS, len(pos_indices)),
            replace=False,
        )

        edges = []
        for idx in chosen:
            src = train_edges[0, idx].item()
            dst = train_edges[1, idx].item()
            edges.append((src, dst))
        return edges

    # ─── Run unlearning (SISA) ───────────────────────────────────────────
    def unlearn_sisa(self, deletion_edges):
        """
        Execute unlearning for SISA-based strategies.
        Returns (avg_time_per_deletion, total_time, certificates).
        """
        self.certificates = []
        total_time = 0.0

        for i, (src, dst) in enumerate(deletion_edges):
            # Hash before
            sid = self.strategy.drug_to_shard.get(src, 0)
            model = self.strategy.models.get(sid)
            hash_before = self.hash_model(model) if model else "no_model"

            # Unlearn
            sid_ret, elapsed = self.strategy.unlearn_edge(src, dst)
            total_time += elapsed

            # Hash after
            model_after = self.strategy.models.get(sid_ret)
            hash_after = self.hash_model(model_after) if model_after else "no_model"

            cert = {
                "deletion_id": i,
                "edge": [src, dst],
                "shard_id": sid_ret,
                "hash_before": hash_before,
                "hash_after": hash_after,
                "retrain_time_s": round(elapsed, 4),
                "strategy": self.strategy.name,
            }
            self.certificates.append(cert)

        avg_time = total_time / max(len(deletion_edges), 1)
        return avg_time, total_time, self.certificates

    # ─── Run unlearning (Gradient Ascent) ────────────────────────────────
    def unlearn_gradient_ascent(self, deletion_edges, data):
        """
        Execute gradient-ascent unlearning (batch mode).
        Returns (avg_time_per_deletion, total_time, certificates).
        """
        self.certificates = []

        # Hash before
        hash_before = self.hash_model(self.strategy.model)

        # Convert to tensor — batch unlearning
        srcs = [e[0] for e in deletion_edges]
        dsts = [e[1] for e in deletion_edges]
        forget_edges = torch.tensor([srcs, dsts], dtype=torch.long)

        total_time = self.strategy.unlearn_edges(forget_edges)

        # Hash after
        hash_after = self.hash_model(self.strategy.model)

        avg_time = total_time / max(len(deletion_edges), 1)

        # One certificate per edge
        for i, (src, dst) in enumerate(deletion_edges):
            cert = {
                "deletion_id": i,
                "edge": [src, dst],
                "shard_id": "global",
                "hash_before": hash_before,
                "hash_after": hash_after,
                "avg_time_per_del_s": round(avg_time, 6),
                "strategy": "GradientAscent",
            }
            self.certificates.append(cert)

        return avg_time, total_time, self.certificates

    # ─── Save certificates ───────────────────────────────────────────────
    def save_certificates(self, strategy_name=None):
        """Write all certificates to JSON files."""
        sname = strategy_name or self.strategy.name
        out_dir = os.path.join(config.CERT_DIR, sname.lower())
        os.makedirs(out_dir, exist_ok=True)

        for cert in self.certificates:
            path = os.path.join(out_dir, f"cert_{cert['deletion_id']:04d}.json")
            with open(path, "w") as f:
                json.dump(cert, f, indent=2)

        # Also save a combined file
        combined_path = os.path.join(config.CERT_DIR, f"{sname.lower()}_all_certs.json")
        with open(combined_path, "w") as f:
            json.dump(self.certificates, f, indent=2)

        return combined_path
