# Study on Adaptation for FiloPriori Integration into Motorola's Production Test Pipeline

**Project:** Filo-Priori V9
**Objective:** Intelligent test case prioritization using Deep Learning
**Status:** Ready for integration with adaptations

---

## 1. INPUT DATA FORMAT REQUIREMENTS

Filo-Priori requires data in the following CSV format:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `Build_ID` | string | **Yes** | Unique build identifier |
| `TC_Key` | string | **Yes** | Test case identifier |
| `TE_Summary` | string | **Yes** | Test description/summary (used for semantic embeddings) |
| `TC_Steps` | string | No | Detailed test steps |
| `TE_Test_Result` | string | **Yes** | `"Pass"` or `"Fail"` |
| `commit` | string | No | Associated commit message (Python list format) |
| `Build_Test_Start_Date` | datetime | No | Build timestamp |

**Example record:**
```csv
Build_ID,TC_Key,TE_Summary,TC_Steps,TE_Test_Result,commit
QPW30.18,MCA-1092,"TE - TC - OTA: Download upgrade package...","1. Check system update...",Pass,"['fix: resolve OTA issue']"
```

**Required adaptation:** Create an ETL (Extract-Transform-Load) module to convert data from Motorola's Android pipeline to this format.

---

## 2. OUTPUT FORMAT (Prioritization)

The system generates two main output files:

### a) Prioritized Test Cases
**File:** `prioritized_test_cases_FULL_testcsv.csv`

```csv
Build_ID,TC_Key,rank,p_fail,verdict
QPW30.18,MCA-1092,1,0.87,Fail
QPW30.18,MCA-1093,2,0.72,Pass
QPW30.18,MCA-1094,3,0.65,Pass
```

| Field | Description |
|-------|-------------|
| `rank` | Position in execution order (1 = highest priority) |
| `p_fail` | Failure probability predicted by the model |
| `verdict` | Actual result (for validation) |

### b) APFD Metrics per Build
**File:** `apfd_per_build_FULL_testcsv.csv`

```csv
method_name,build_id,test_scenario,count_tc,count_commits,apfd
filo_priori_v9,QPW30.18,regression,45,12,0.82
filo_priori_v9,QPW30.19,smoke,23,5,0.91
```

**Integration:** The `rank` output can be used directly to order test execution in the CI/CD pipeline.

---

## 3. INFERENCE PIPELINE (Production)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FILO-PRIORI INFERENCE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. DATA INPUT                                                          │
│     └── CSV with Build_ID, TC_Key, TE_Summary, TE_Test_Result           │
│                                                                         │
│  2. SEMANTIC STREAM (SBERT)                                             │
│     ├── Model: sentence-transformers/all-mpnet-base-v2                  │
│     ├── Input: TE_Summary + TC_Steps + commit                           │
│     ├── Output: 1536-dim embedding (768 TC + 768 Commit)                │
│     └── Cache: Reuses embeddings for previously seen tests              │
│                                                                         │
│  3. STRUCTURAL STREAM (GAT)                                             │
│     ├── Features: 19 dimensions extracted from history                  │
│     │   ├── failure_rate, recent_failure_rate, flakiness                │
│     │   ├── consecutive_failures, test_age, test_novelty                │
│     │   └── execution_status_last_[1,2,3,5,10], cycles_since_fail       │
│     ├── Graph: Multi-edge (co-failure, semantic, temporal, component)   │
│     └── Cold-start: KNN + regressor for new tests                       │
│                                                                         │
│  4. FUSION & PREDICTION                                                 │
│     ├── Cross-Attention: 4 heads, bidirectional                         │
│     ├── Classifier: MLP [512 → 128 → 64 → 2]                            │
│     └── Output: P(Fail) for each test                                   │
│                                                                         │
│  5. RANKING                                                             │
│     ├── Sort by P(Fail) descending                                      │
│     ├── Optimized threshold: 0.2777                                     │
│     └── Orphan handling: KNN blend for tests without history            │
│                                                                         │
│  INFERENCE TIME: < 1 second per build                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Execution Command
```bash
python main.py --config configs/experiment_industry_optimized_v3.yaml
```

---

## 4. TECHNICAL REQUIREMENTS

### Hardware

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | - | NVIDIA 8GB+ VRAM (CUDA 11.8+) |
| RAM | 16 GB | 32 GB |
| Disk | 5 GB | 10 GB (embedding cache) |
| CPU | 4 cores | 8+ cores |

### Software

```
Python 3.12+
PyTorch >= 2.0.0
PyTorch Geometric >= 2.3.0
sentence-transformers >= 2.2.2
transformers >= 4.30.0
scikit-learn >= 1.3.0
pandas >= 2.0.0
numpy >= 1.24.0
networkx >= 3.0
PyYAML >= 6.0
CUDA 11.8+ (optional but recommended)
```

### Execution Time

| Operation | Time | Resource |
|-----------|------|----------|
| Inference (per build) | < 1 second | GPU/CPU |
| Full training | 2-3 hours | GPU |
| Initial embedding generation | 15-30 min | GPU |
| Graph construction | 5-15 min | CPU |
| Model size | 9.5 MB | Disk |

---

## 5. ADAPTATION POINTS FOR MOTOROLA PIPELINE

### 5.1 Data Adapter Module (NEW)

```python
# Required: src/adapters/motorola_adapter.py

import pandas as pd
from typing import Dict, List

class MotorolaDataAdapter:
    """
    Adapter to convert Motorola pipeline data
    to Filo-Priori format.
    """

    # Motorola → Filo-Priori field mapping
    FIELD_MAPPING = {
        'android_build_id': 'Build_ID',
        'test_case_id': 'TC_Key',
        'test_description': 'TE_Summary',
        'test_steps': 'TC_Steps',
        'result': 'TE_Test_Result',
        'git_commit_sha': 'commit',
        'build_timestamp': 'Build_Test_Start_Date'
    }

    # Result value mapping
    RESULT_MAPPING = {
        'PASSED': 'Pass',
        'FAILED': 'Fail',
        'SUCCESS': 'Pass',
        'FAILURE': 'Fail',
        'pass': 'Pass',
        'fail': 'Fail'
    }

    def convert_to_filopriori_format(
        self,
        motorola_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert DataFrame from Motorola format to Filo-Priori.

        Args:
            motorola_data: DataFrame with Motorola pipeline data

        Returns:
            DataFrame in Filo-Priori expected format
        """
        df = motorola_data.copy()

        # Rename columns
        df = df.rename(columns=self.FIELD_MAPPING)

        # Convert result values
        df['TE_Test_Result'] = df['TE_Test_Result'].map(
            self.RESULT_MAPPING
        ).fillna(df['TE_Test_Result'])

        # Ensure required columns
        required = ['Build_ID', 'TC_Key', 'TE_Summary', 'TE_Test_Result']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Required column missing: {col}")

        # Fill optional columns with empty values
        if 'TC_Steps' not in df.columns:
            df['TC_Steps'] = ""
        if 'commit' not in df.columns:
            df['commit'] = "[]"

        return df

    def validate_data(self, df: pd.DataFrame) -> Dict:
        """Validate converted data and return statistics."""
        stats = {
            'total_records': len(df),
            'unique_builds': df['Build_ID'].nunique(),
            'unique_tests': df['TC_Key'].nunique(),
            'pass_count': (df['TE_Test_Result'] == 'Pass').sum(),
            'fail_count': (df['TE_Test_Result'] == 'Fail').sum(),
            'missing_summary': df['TE_Summary'].isna().sum()
        }
        return stats
```

### 5.2 CI/CD Integration Points

```yaml
# Jenkins integration example
# Jenkinsfile or pipeline script

stages:
  - name: prioritize_tests
    script:
      - python scripts/motorola_prioritize.py \
          --input ${TEST_CASES_CSV} \
          --output ${PRIORITIZED_OUTPUT} \
          --model results/experiment_industry_optimized_v3/best_model.pt \
          --config configs/experiment_motorola.yaml
    artifacts:
      paths:
        - prioritized_order.json
```

```yaml
# GitHub Actions example
name: Test Prioritization

on:
  push:
    branches: [main, develop]

jobs:
  prioritize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run Filo-Priori
        run: |
          python main.py \
            --config configs/experiment_motorola.yaml \
            --inference-only \
            --input ${{ github.workspace }}/test_cases.csv

      - name: Upload prioritized tests
        uses: actions/upload-artifact@v3
        with:
          name: prioritized-tests
          path: results/prioritized_test_cases.csv
```

### 5.3 Configuration Adjustments

```yaml
# configs/experiment_motorola.yaml (to be created)

experiment:
  name: "experiment_motorola_android"
  version: "1.0"
  description: "Motorola Android Test Pipeline"
  seed: 42

# Data Configuration - ADJUST PATHS
data:
  train_path: "datasets/motorola/train.csv"
  test_path: "datasets/motorola/test.csv"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  random_seed: 42
  binary_classification: true
  binary_strategy: "pass_vs_fail"
  binary_positive_class: "Pass"
  binary_negative_class: "Fail"

# Embedding - Separate cache for Motorola
embedding:
  model_name: "sentence-transformers/all-mpnet-base-v2"
  embedding_dim: 768
  combined_embedding_dim: 1536
  max_length: 384
  batch_size: 16
  normalize_embeddings: true
  device: "cuda"  # or "cpu" if no GPU
  use_cache: true
  cache_dir: "cache/motorola"

# Graph - Separate cache
graph:
  build_graph: true
  use_multi_edge: true
  edge_types: [co_failure, co_success, semantic, temporal, component]
  cache_path: "cache/motorola/multi_edge_graph.pkl"

# Model - Keep validated architecture
model:
  type: "dual_stream"
  num_classes: 2
  # ... (keep v3 configuration)

# Output
output:
  results_dir: "results/motorola"
  save_predictions: true
  save_rankings: true
  save_apfd: true
```

---

## 6. COLD-START HANDLING (New Tests)

The system handles **22.7% orphan tests** (without history) through a 4-stage pipeline:

| Stage | Strategy | Parameter |
|-------|----------|-----------|
| 1 | KNN Similarity | k=20 nearest neighbors |
| 2 | Structural Blend | 35% weight for structural features |
| 3 | Temperature Scaling | T=0.7 for softmax |
| 4 | Alpha Blend | 55% KNN + 45% base score |

### Impact
- **Before:** Orphan variance = 0.0 (all received same score)
- **After:** Orphan variance = 0.0462 (effective differentiation)
- **APFD Improvement:** +5.9% on builds with new tests

### Relevant Code
```python
# src/evaluation/orphan_ranker.py
def compute_orphan_scores(
    orphan_indices,
    train_embeddings,
    train_structural,
    train_scores,
    k_neighbors=20,
    alpha_blend=0.55,
    temperature=0.7
):
    """Compute scores for tests without history via KNN."""
```

---

## 7. RECOMMENDED INTEGRATION PHASES

| Phase | Description | Deliverables |
|-------|-------------|--------------|
| **1. Data Mapping** | Create adapter for Motorola → Filo-Priori format | `motorola_adapter.py`, field documentation |
| **2. Validation** | Run current model with converted Motorola data | APFD report, error analysis |
| **3. Fine-tuning** | Adjust hyperparameters for Android patterns | Optimized `experiment_motorola.yaml` |
| **4. Retraining** | (Optional) Train with Motorola historical data | New `best_model.pt` |
| **5. Staging** | Deploy to staging environment | Integration scripts, E2E tests |
| **6. Production** | Full integration into CI/CD pipeline | Automated pipeline |

---

## 8. API INTERFACE (Proposal)

```python
# Proposal: src/api/filopriori_service.py

import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path

class FiloPrioriService:
    """
    Test prioritization service for CI/CD pipeline integration.
    """

    def __init__(
        self,
        model_path: str = "results/experiment_industry_optimized_v3/best_model.pt",
        config_path: str = "configs/experiment_motorola.yaml"
    ):
        """
        Initialize the service by loading model and configuration.

        Args:
            model_path: Path to the trained model
            config_path: Path to configuration file
        """
        self.model = self._load_model(model_path)
        self.config = self._load_config(config_path)
        self.embedding_cache = {}

    def prioritize_build(
        self,
        build_id: str,
        test_cases: List[Dict],
        return_format: str = "list"
    ) -> List[Dict]:
        """
        Prioritize test cases for a specific build.

        Args:
            build_id: Build identifier
            test_cases: List of dictionaries with test information
                [{"tc_key": "...", "summary": "...", "steps": "..."}]
            return_format: "list" or "dataframe"

        Returns:
            List ordered by priority:
            [
                {"tc_key": "MCA-1092", "rank": 1, "p_fail": 0.87},
                {"tc_key": "MCA-1093", "rank": 2, "p_fail": 0.72},
                ...
            ]
        """
        # Implementation
        pass

    def prioritize_batch(
        self,
        builds: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """
        Prioritize multiple builds in batch.

        Args:
            builds: {build_id: [test_cases]}

        Returns:
            {build_id: [prioritized_tests]}
        """
        pass

    def update_history(
        self,
        execution_results: pd.DataFrame
    ) -> None:
        """
        Update history with execution results.
        Used to keep structural features up to date.

        Args:
            execution_results: DataFrame with Build_ID, TC_Key, TE_Test_Result
        """
        pass

    def get_model_info(self) -> Dict:
        """Return information about the loaded model."""
        return {
            "model_version": "v9",
            "apfd_mean": 0.7595,
            "threshold": 0.2777,
            "embedding_model": "all-mpnet-base-v2"
        }
```

### Usage Example

```python
from src.api.filopriori_service import FiloPrioriService

# Initialize service
service = FiloPrioriService(
    model_path="results/experiment_industry_optimized_v3/best_model.pt",
    config_path="configs/experiment_motorola.yaml"
)

# Prepare tests
test_cases = [
    {
        "tc_key": "MOTO-001",
        "summary": "Verify camera app launches correctly",
        "steps": "1. Open camera app\n2. Check preview"
    },
    {
        "tc_key": "MOTO-002",
        "summary": "Test WiFi connection stability",
        "steps": "1. Connect to WiFi\n2. Run ping test"
    }
]

# Get prioritization
prioritized = service.prioritize_build(
    build_id="ANDROID-14-BUILD-001",
    test_cases=test_cases
)

# Result
# [
#     {"tc_key": "MOTO-002", "rank": 1, "p_fail": 0.73},
#     {"tc_key": "MOTO-001", "rank": 2, "p_fail": 0.45}
# ]
```

---

## 9. RISKS & MITIGATIONS

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Different data format** | High | High | Create robust adapter with validation and detailed logs |
| **Concept drift** | Medium | Medium | Monitor APFD weekly, retrain if < 0.70 |
| **GPU latency** | Low | Low | Supports CPU (slower but functional), use batch |
| **Android-specific tests** | Medium | Medium | Fine-tuning with Motorola data, analyze embeddings |
| **Cold-start on releases** | High | Medium | KNN pipeline already implemented, monitor orphan % |
| **Scalability** | Low | Medium | Embedding cache, < 1s inference already validated |

### Recommended Monitoring

```python
# Metrics to monitor in production
metrics = {
    "apfd_mean": 0.75,      # Alert if < 0.70
    "apfd_min": 0.50,       # Alert if < 0.40
    "orphan_ratio": 0.25,   # Alert if > 0.35
    "inference_time_ms": 1000,  # Alert if > 2000
    "cache_hit_rate": 0.80  # Alert if < 0.60
}
```

---

## 10. DELIVERABLES FOR INTEGRATION

### Already Available

| Item | Location | Status |
|------|----------|--------|
| Trained model | `results/experiment_industry_optimized_v3/best_model.pt` | Ready |
| Validated configuration | `configs/experiment_industry_optimized_v3.yaml` | Ready |
| Training/inference pipeline | `main.py` | Ready |
| Technical documentation | `docs/TECHNICAL_REPORT_APFD_0.7595.md` | Ready |
| APFD calculation | `src/evaluation/apfd.py` | Ready |
| Cold-start handler | `src/evaluation/orphan_ranker.py` | Ready |

### To Be Developed

| Item | Description | Priority |
|------|-------------|----------|
| `src/adapters/motorola_adapter.py` | Motorola → Filo-Priori data conversion | High |
| `configs/experiment_motorola.yaml` | Motorola-specific configuration | High |
| `src/api/filopriori_service.py` | Service API for integration | Medium |
| CI/CD integration scripts | Jenkins/GitHub Actions | Medium |
| Validation tests | E2E tests with Motorola data | High |
| Monitoring dashboard | Production metrics | Low |

---

## 11. PERFORMANCE SUMMARY

### Validated Results (Industrial Dataset)

| Metric | Value |
|--------|-------|
| **Mean APFD** | 0.7595 |
| **Median APFD** | 0.7944 |
| **APFD ≥ 0.7** | 67.9% of builds |
| **APFD ≥ 0.5** | 89.2% of builds |
| **Validated builds** | 277 |

### Baseline Comparison

| Method | APFD | vs Filo-Priori |
|--------|------|----------------|
| **Filo-Priori V9** | **0.7595** | - |
| DeepOrder | 0.6892 | -9.3% |
| NodeRank | 0.6630 | -12.7% |
| FailureRate | 0.6289 | -17.2% |
| Random | 0.5596 | -26.3% |

---

## 12. CONTACT & SUPPORT

**Team:** UFAM/IATS
**Project:** Filo-Priori V9
**Repository:** `filo_priori_v9/`

For technical questions, consult:
- `docs/TECHNICAL_REPORT_APFD_0.7595.md` - Complete technical report
- `docs/PIPELINE_ARCHITECTURE.md` - Visual architecture
- `paper/APRESENTACAO_PREPARACAO.md` - Prepared Q&A

---

*Document generated: January 2026*
*Version: 1.0*
