"""
CEUBG — Random Sharding Strategy.

Uniformly assign drug nodes to K shards via random permutation.
All edges for a drug follow it to its shard.
"""
import numpy as np
from base_shard import BaseShard
import config


class RandomShard(BaseShard):
    """Uniform random drug-node assignment (baseline)."""

    def assign_shards(self):
        rng = np.random.default_rng(config.UNLEARN_SEED)
        perm = rng.permutation(config.NUM_DRUG_NODES)
        chunk = config.NUM_DRUG_NODES // self.k

        self.shard_to_drugs = {i: [] for i in range(self.k)}
        self.drug_to_shard = {}

        for i, drug_id in enumerate(perm):
            sid = min(i // chunk, self.k - 1)
            self.drug_to_shard[drug_id] = sid
            self.shard_to_drugs[sid].append(drug_id)

    @property
    def name(self):
        return "Random"
