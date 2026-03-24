"""
CEUBG — Similarity Sharding Strategy (primary).

TruncatedSVD(256) on sparse Morgan fingerprints → L2-normalize → KMeans(K=20).
Route each test edge to its SINGLE owning shard — no averaging across shards.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

from base_shard import BaseShard
import config


class SimilarityShard(BaseShard):
    """
    Similarity-based sharding via chemical fingerprint clustering.
    TruncatedSVD (NOT PCA — preserves binary sparsity) → KMeans.
    """

    def assign_shards(self):
        # Load Morgan fingerprints
        fp_df = pd.read_csv(config.FINGERPRINTS_PATH, index_col=0)
        fp_matrix = csr_matrix(fp_df.values.astype(np.float32))

        # TruncatedSVD — NOT PCA (centering destroys binary sparsity)
        svd = TruncatedSVD(
            n_components=config.SVD_COMPONENTS,
            random_state=config.KMEANS_SEED,
        )
        X_reduced = svd.fit_transform(fp_matrix)

        # L2-normalize
        X_normed = normalize(X_reduced, norm="l2")

        # KMeans clustering
        kmeans = KMeans(
            n_clusters=self.k,
            n_init=config.KMEANS_N_INIT,
            max_iter=config.KMEANS_MAX_ITER,
            random_state=config.KMEANS_SEED,
        )
        cluster_ids = kmeans.fit_predict(X_normed)

        # Map drug index → shard
        self.shard_to_drugs = {i: [] for i in range(self.k)}
        self.drug_to_shard = {}

        num_available = min(len(cluster_ids), config.NUM_DRUG_NODES)
        for drug_id in range(num_available):
            sid = int(cluster_ids[drug_id])
            self.drug_to_shard[drug_id] = sid
            self.shard_to_drugs[sid].append(drug_id)

        # Any remaining drug nodes (if fingerprint file is shorter) → shard 0
        for drug_id in range(num_available, config.NUM_DRUG_NODES):
            self.drug_to_shard[drug_id] = 0
            self.shard_to_drugs[0].append(drug_id)

        # Save mapping
        map_path = os.path.join(config.OUTPUT_DIR, "drug_shard_map_similarity.json")
        with open(map_path, "w") as f:
            json.dump({str(k): v for k, v in self.drug_to_shard.items()}, f)

    @property
    def name(self):
        return "Similarity"
