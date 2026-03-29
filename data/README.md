# Data Directory

This directory contains the datasets used for training and evaluation.

## Structure

```
data/
├── raw/           # Original downloaded datasets
├── processed/     # Preprocessed graph data
└── features/      # Extracted features (fingerprints, embeddings)
```

## Datasets

- **ogbl-ddi**: Drug-Drug Interaction dataset from Open Graph Benchmark
- **KIBA**: Drug-Target Interaction dataset

## Note

Raw data files are gitignored. Use the data loading scripts in `src/data/` to download and preprocess datasets.
