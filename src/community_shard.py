"""
CEUBG — Community Sharding Strategy.

Louvain community detection on the DTI graph, merged into 20 balanced shards.
Uses Configuration A (coarser grouping).
"""
import numpy as np
import networkx as nx

try:
    import community as community_louvain   # python-louvain
except ImportError:
    community_louvain = None

from base_shard import BaseShard
import config


class CommunityShard(BaseShard):
    """
    Louvain community detection on positive DTI edges.
    Merge tiny communities and split large ones to produce exactly K balanced shards.
    Configuration A — coarser grouping.
    """

    def assign_shards(self):
        if community_louvain is None:
            raise ImportError(
                "python-louvain is required:  pip install python-louvain"
            )

        # Build undirected graph from POSITIVE train edges
        G = nx.Graph()
        train_edges  = self.data["train_edges"]
        train_labels = self.data["train_labels"]

        for idx in range(train_edges.shape[1]):
            if train_labels[idx].item() == 1.0:
                src = train_edges[0, idx].item()
                dst = train_edges[1, idx].item()
                G.add_edge(src, dst)

        # Add any isolated drug nodes so they aren't lost
        for d in range(config.NUM_DRUG_NODES):
            if d not in G:
                G.add_node(d)

        # Louvain detection (resolution=1.0 for coarser grouping — Config A)
        partition = community_louvain.best_partition(
            G, resolution=1.0, random_state=config.UNLEARN_SEED
        )

        # Collect raw communities (may be >> K or << K)
        raw_comms = {}
        for node, comm_id in partition.items():
            if node < config.NUM_DRUG_NODES:  # only drug nodes
                raw_comms.setdefault(comm_id, []).append(node)

        # Sort communities by size descending
        sorted_comms = sorted(raw_comms.values(), key=len, reverse=True)

        # Merge / split to get exactly K balanced shards
        target_size = config.NUM_DRUG_NODES // self.k
        shards = [[] for _ in range(self.k)]

        # Greedy best-fit: assign each community to the shard with fewest members
        for comm in sorted_comms:
            if len(comm) > target_size * 2:
                # Split large community
                np.random.RandomState(42).shuffle(comm)
                chunks = [comm[i:i + target_size] for i in range(0, len(comm), target_size)]
                for chunk in chunks:
                    # find smallest shard
                    sid = min(range(self.k), key=lambda s: len(shards[s]))
                    shards[sid].extend(chunk)
            else:
                sid = min(range(self.k), key=lambda s: len(shards[s]))
                shards[sid].extend(comm)

        # Build maps
        self.shard_to_drugs = {}
        self.drug_to_shard = {}
        assigned = set()
        for sid in range(self.k):
            self.shard_to_drugs[sid] = shards[sid]
            for drug_id in shards[sid]:
                self.drug_to_shard[drug_id] = sid
                assigned.add(drug_id)

        # Ensure all drug nodes are assigned
        for d in range(config.NUM_DRUG_NODES):
            if d not in assigned:
                sid = min(range(self.k), key=lambda s: len(self.shard_to_drugs[s]))
                self.drug_to_shard[d] = sid
                self.shard_to_drugs[sid].append(d)

    @property
    def name(self):
        return "Community"
