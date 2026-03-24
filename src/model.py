"""
CEUBG — Two-layer GraphSAGE model.

Message passing uses the full (global) graph for structural richness.
Edge score = dot(drug_emb, protein_emb).  BCE loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEModel(nn.Module):
    """Two-layer GraphSAGE encoder + dot-product link predictor."""

    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    # ── Encoder ──────────────────────────────────────────────────────────
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Produce node embeddings using the GLOBAL edge_index for message passing.
        """
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h

    # ── Link predictor ───────────────────────────────────────────────────
    @staticmethod
    def predict(z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        Edge score = dot product of source and target embeddings.

        Args:
            z: (N, D) node embeddings
            edges: (2, E) edge index pairs
        Returns:
            (E,) predicted scores (logits)
        """
        src, dst = edges[0], edges[1]
        return (z[src] * z[dst]).sum(dim=1)

    def forward(self, x, edge_index, pred_edges):
        """Convenience: encode then predict."""
        z = self.encode(x, edge_index)
        return self.predict(z, pred_edges)
