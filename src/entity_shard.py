"""
CEUBG — Entity Sharding Strategy.

Partition drug nodes into K equal contiguous index-range buckets.
Drug indices 0–2623 → shard 0, 2624–5247 → shard 1, etc.
"""
from base_shard import BaseShard
import config


class EntityShard(BaseShard):
    """Group drugs by KIBA index ranges (contiguous bucketing)."""

    def assign_shards(self):
        chunk = config.NUM_DRUG_NODES // self.k

        self.shard_to_drugs = {i: [] for i in range(self.k)}
        self.drug_to_shard = {}

        for drug_id in range(config.NUM_DRUG_NODES):
            sid = min(drug_id // chunk, self.k - 1)
            self.drug_to_shard[drug_id] = sid
            self.shard_to_drugs[sid].append(drug_id)

    @property
    def name(self):
        return "Entity"
