"""
CEUBG — Central configuration for all hyperparameters, paths, and constants.
"""
import os
import torch

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

NODE_FEATURES_PATH  = os.path.join(BASE_DIR, "node_features.npy")
EDGE_INDEX_PATH     = os.path.join(BASE_DIR, "edge_index.npy")
EDGE_LABELS_PATH    = os.path.join(BASE_DIR, "edge_labels.npy")
TRAIN_EDGES_PATH    = os.path.join(BASE_DIR, "train_edges.npy")
VAL_EDGES_PATH      = os.path.join(BASE_DIR, "val_edges.npy")
TEST_EDGES_PATH     = os.path.join(BASE_DIR, "test_edges.npy")
NODE_MAPPING_PATH   = os.path.join(BASE_DIR, "node_mapping.json")
FINGERPRINTS_PATH   = os.path.join(DATA_DIR, "drug_morgan_fingerprints_after_fp_clean.csv")
EDGES_CSV_PATH      = os.path.join(DATA_DIR, "kiba_edges_balanced.csv")

OUTPUT_DIR          = os.path.join(BASE_DIR, "results")
CERT_DIR            = os.path.join(OUTPUT_DIR, "certificates")
PLOTS_DIR           = os.path.join(OUTPUT_DIR, "plots")

# ─── Dataset constants ───────────────────────────────────────────────────
NUM_DRUG_NODES    = 52_477
NUM_PROTEIN_NODES = 467
NUM_TOTAL_NODES   = NUM_DRUG_NODES + NUM_PROTEIN_NODES  # 52,944
NODE_FEATURE_DIM  = 2048
NUM_POS_EDGES     = 59_852
NUM_TOTAL_EDGES   = 119_704  # balanced (pos + neg)

# ─── GNN hyperparams ─────────────────────────────────────────────────────
HIDDEN_DIM    = 128
NUM_GNN_LAYERS = 2
LEARNING_RATE = 0.01
WEIGHT_DECAY  = 1e-5
EPOCHS        = 200
PATIENCE      = 20   # early stopping patience

# ─── Sharding ─────────────────────────────────────────────────────────────
K_SHARDS = 20

# ─── Unlearning ───────────────────────────────────────────────────────────
NUM_BENCHMARK_DELETIONS = 100
UNLEARN_SEED            = 123

# ─── Gradient Ascent ──────────────────────────────────────────────────────
GA_STEPS     = 10    # fewer steps at very low LR for selective forgetting
GA_LR        = 0.0005 # much smaller than training LR to avoid catastrophic drift
GA_CLIP_NORM = 1.0   # gradient clipping max norm

# ─── Full Retraining Baseline ─────────────────────────────────────────────
FULL_RETRAIN_EPOCHS = EPOCHS  # same as training epochs for fair comparison

# ─── Similarity strategy ─────────────────────────────────────────────────
SVD_COMPONENTS = 256
KMEANS_N_INIT  = 50
KMEANS_MAX_ITER = 500
KMEANS_SEED    = 42

# ─── Device ───────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Ensure output dirs exist ────────────────────────────────────────────
for _d in [OUTPUT_DIR, CERT_DIR, PLOTS_DIR]:
    os.makedirs(_d, exist_ok=True)
