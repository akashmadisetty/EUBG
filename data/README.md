# Data Directory

This directory contains the KiBA (Kinase Inhibitor BioActivity) dataset used for drug-target interaction prediction in the EUBG framework.

## Files

| File | Description |
|------|-------------|
| `KiBA_final_after_fp_clean.xlsx` | Cleaned KiBA bioactivity matrix after fingerprint filtering |
| `drug_morgan_fingerprints_after_fp_clean.csv` | 2048-bit Morgan fingerprint vectors for each drug compound |
| `drug_smiles_kiba_final_after_fp_clean.csv` | SMILES strings for drug compounds in the cleaned KiBA set |
| `kiba_edges_balanced.csv` | Balanced edge list (positive and negative drug-target pairs) |

## Preprocessing

The raw KiBA dataset has been preprocessed as follows:

1. Drug compounds without valid Morgan fingerprints were removed
2. Morgan fingerprints (radius=2, 2048 bits) were computed from SMILES representations
3. Positive interactions were paired with an equal number of negative samples to create a balanced edge set

The data loading pipeline in `src/data_loader.py` reads these files and constructs the graph tensors used for training and evaluation.
