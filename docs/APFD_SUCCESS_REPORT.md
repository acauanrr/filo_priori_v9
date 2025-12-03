# Filo-Priori V3 — APFD 0.7595 Success Report

This document explains why the latest run achieved strong APFD results on the full test.csv (277 builds) and how each stage of the pipeline contributes.

## Key Outcomes
- APFD (277 builds): **0.7595** mean (median 0.7944; 67.9% builds ≥0.7; 89.2% builds ≥0.5)
- Test split: APFD 0.6966; Val F1 0.5899; Test F1 0.5870
- Zero build leakage (grouped splits by `Build_ID`)

## End-to-End Pipeline

### 1) Data Loading & Splitting
- `DataLoader` performs **group-aware splits** by `Build_ID` to prevent leakage (`train/val/test` disjoint builds).
- Non-strict cleaning for **full test.csv** keeps every row to preserve per-build test counts; labels outside Pass/Fail are excluded only when encoding binary labels.

**Inputs:** `train.csv`, `test.csv`  
**Outputs:** `train/val/test` DataFrames + label mapping + class weights

### 2) Text Embeddings (SBERT)
- Model: `sentence-transformers/all-mpnet-base-v2`, concatenating **TC** and **commit** embeddings → 1536-d vectors.
- Intelligent caching prevents recomputation; full-test uses a **separate cache** (`cache/.../full_test`) to avoid train/test cross-talk.

**Inputs:** TC summary, steps, commit text  
**Outputs:** `train/val/test` embedding matrices

### 3) Structural Features + Imputation + DeepOrder
- Extractor V2.5 (10 curated features) → **19 total** after adding **DeepOrder priority features** (9).
- **Imputation** for cold tests uses KNN over train embeddings + structural base features (no label leakage); 22.7% imputed in full test.
- Priority scores computed chronologically (train → val → test; full-test uses independent history).

**Inputs:** Execution history (Build_ID, verdict), component metadata  
**Outputs:** Structural matrix (19-d), priority scores

### 4) Multi-Edge Graph Construction
- Edge types: **co_failure, co_success, semantic, temporal, component**; `semantic_top_k=10`, `semantic_threshold=0.65`, `weight_threshold=0.03`.
- Denser graph reduces orphans (77.4% in-graph coverage on full test).

**Inputs:** Train DataFrame + embeddings  
**Outputs:** `edge_index`, `edge_weights`, TC→global index map

### 5) Model (Dual Stream + GAT)
- Semantic stream (MLP) and structural stream (GAT) with aligned hidden dims.
- Fusion via cross-attention; classifier head for Fail/Pass.
- Balanced sampling (~15:1) stabilizes recall/precision trade-off without extreme bias.

**Inputs:** Batches of embeddings, structural features, subgraph slices  
**Outputs:** Logits/probabilities per batch

### 6) Threshold Optimization
- Two-phase search (coarse 0.05, fine 0.01, `f_beta` with β=0.8) on **validation only** to tune recall/precision.
- Applied during reporting; default remains 0.5 if disabled.

**Inputs:** Val probabilities/labels  
**Outputs:** Optimal threshold + metrics report

### 7) Orphan Handling (High-Variance KNN)
- Shared scorer (`evaluation/orphan_ranker.py`): temperature-scaled KNN (k=20, euclidean), blends semantic (65%) + structural (35%) similarity, uses priority fallback.
- Applied in both the test split and full-test ranking. Delivered orphan stats: mean 0.3717, std 0.0462 (no flat 0.2011 plateaus).

**Inputs:** Orphan embeddings, in-graph scores, structural features, priority fallback  
**Outputs:** Orphan scores with healthy variance

### 8) Ranking & APFD
- Hybrid ranking: P(Fail) primary; orphan KNN boosts missing scores; APFD computed per build.
- Outputs stored under `results/experiment_industry_optimized_v3/`:
  - `prioritized_test_cases.csv` (test split)
  - `apfd_per_build.csv` (test split)
  - `prioritized_test_cases_FULL_testcsv.csv` (277 builds)
  - `apfd_per_build_FULL_testcsv.csv` (277 builds)

## What Drove the APFD Gain
1) **Denser multi-edge graph** (semantic_top_k=10, lower threshold, temporal/component edges) → fewer orphans, better message passing.  
2) **High-variance orphan scorer** (k=20, euclidean, structural blend, temperature) → eliminated flat scores; orphans now differentiated.  
3) **Balanced sampling and tuned threshold search** (two-phase, `f_beta` 0.8) → improved early-fail capture without over-predicting Fail.  
4) **DeepOrder features + priority history** → informative structural priors, especially for rarely failing tests.  
5) **Strict build-level split & isolated caches** → no leakage; metrics reflect genuine generalization.

## Inputs and Outputs Summary
- **Inputs:** `datasets/01_industry/train.csv`, `datasets/01_industry/test.csv`, cached embeddings, structural cache, graph cache.  
- **Intermediate outputs:** embeddings (`cache/...`), structural cache, graph cache (`cache/01_industry/multi_edge_graph.pkl`).  
- **Final outputs:** probabilities, hybrid scores, prioritized CSVs, APFD reports in `results/experiment_industry_optimized_v3/`.

## Suggested Next Checks
- Keep the build-overlap assertion on splits (already logged).
- If running on new data, refresh full-test cache (`--force-regen-embeddings`) to avoid stale embedding drift.
- Tune `ranking.orphan_strategy.k_neighbors` (default 20) for new datasets with different orphan ratios.

## How to Reproduce the Run
1) (Optional) Regenerate embeddings to avoid stale cache:
   - `python main.py --config configs/experiment_industry_optimized_v3.yaml --force-regen-embeddings`
2) Standard run (uses cached embeddings if valid):
   - `python main.py --config configs/experiment_industry_optimized_v3.yaml`
3) Outputs of interest (written automatically):
   - `results/experiment_industry_optimized_v3/prioritized_test_cases.csv` (test split)
   - `results/experiment_industry_optimized_v3/apfd_per_build.csv` (test split)
   - `results/experiment_industry_optimized_v3/prioritized_test_cases_FULL_testcsv.csv` (277 builds)
   - `results/experiment_industry_optimized_v3/apfd_per_build_FULL_testcsv.csv` (277 builds)

### Quick Visuals (optional)
You can plot APFD distribution directly from the saved CSVs. Example:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/experiment_industry_optimized_v3/apfd_per_build_FULL_testcsv.csv")
df['APFD'].hist(bins=20, range=(0,1))
plt.title("APFD Distribution (277 builds)")
plt.xlabel("APFD"); plt.ylabel("Build count")
plt.show()
```

To inspect orphan score variance:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("results/experiment_industry_optimized_v3/prioritized_test_cases_FULL_testcsv.csv")
orphans = df[df['probability'].sub(0.5).abs() < 1e-3]
print(f"Orphans: {len(orphans)}")
print(orphans['hybrid_score'].describe(percentiles=[0.1,0.5,0.9]))
```
