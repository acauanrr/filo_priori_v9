#!/bin/bash
# =============================================================================
# Filo-Priori vs Baselines - RTPTorrent Dataset (02_rtptorrent) Experiment
# =============================================================================
# Este script executa o experimento completo no dataset 02_rtptorrent:
# 1. Treina e avalia o modelo Filo-Priori (Learning-to-Rank)
# 2. Compara contra 7 baselines do RTPTorrent:
#    - untreated: Ordem original do build log
#    - random: Ordem aleatoria
#    - recently-failed: Ordenado por historico de falhas recentes
#    - optimal-failure: Ordenacao otima (falhas primeiro) - upper bound
#    - optimal-failure-duration: Otima (falhas mais curtas primeiro)
#    - matrix-naive: Matriz arquivo-teste-falhas
#    - matrix-conditional-prob: Probabilidade condicional P(teste | arquivos_alterados)
#
# Uso:
#   ./run_rtptorrent_experiment.sh
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="configs/experiment_rtptorrent_l2r.yaml"
RESULTS_BASE="results/rtptorrent_comparison_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RESULTS_BASE}/experiment.log"

# Create results directory
mkdir -p "${RESULTS_BASE}"

# Function to log messages
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${LOG_FILE}"
}

log_header() {
    echo "" | tee -a "${LOG_FILE}"
    echo -e "${YELLOW}============================================================${NC}" | tee -a "${LOG_FILE}"
    echo -e "${YELLOW}$1${NC}" | tee -a "${LOG_FILE}"
    echo -e "${YELLOW}============================================================${NC}" | tee -a "${LOG_FILE}"
}

# =============================================================================
# HEADER
# =============================================================================
echo ""
echo "============================================================"
echo "   FILO-PRIORI vs BASELINES - RTPTorrent Dataset Experiment"
echo "============================================================"
echo "Config: ${CONFIG_FILE}"
echo "Results: ${RESULTS_BASE}"
echo "Started: $(date)"
echo ""
echo "Baselines compared:"
echo "  1. untreated (original order)"
echo "  2. random"
echo "  3. recently-failed"
echo "  4. optimal-failure (upper bound)"
echo "  5. optimal-failure-duration"
echo "  6. matrix-naive"
echo "  7. matrix-conditional-prob"
echo "============================================================"
echo ""

log "Experiment started"
log "Config: ${CONFIG_FILE}"
log "Results directory: ${RESULTS_BASE}"

# =============================================================================
# CHECK DATASET
# =============================================================================
log_header "STEP 0: Checking Dataset"

if [ ! -d "datasets/02_rtptorrent/processed_ranking" ]; then
    log_error "Processed ranking data not found!"
    log "Please run: python scripts/preprocessing/preprocess_rtptorrent_ranking.py --preset small"

    # Check if raw data exists
    if [ -d "datasets/02_rtptorrent/raw/MSR2" ]; then
        log "Raw data found. Attempting to preprocess..."

        # Check if preprocessing script exists (it was removed, so we need to handle this)
        if [ -f "scripts/preprocessing/preprocess_rtptorrent_ranking.py" ]; then
            python scripts/preprocessing/preprocess_rtptorrent_ranking.py --preset small
        else
            log_error "Preprocessing script not found. Dataset needs to be prepared manually."
            log "Expected: datasets/02_rtptorrent/processed_ranking/train.csv"
            log "Expected: datasets/02_rtptorrent/processed_ranking/test.csv"
            exit 1
        fi
    else
        log_error "Raw data not found at datasets/02_rtptorrent/raw/MSR2"
        log "Please download the RTPTorrent dataset first"
        exit 1
    fi
else
    log_success "Dataset found at datasets/02_rtptorrent/processed_ranking"
fi

# Show dataset info
if [ -f "datasets/02_rtptorrent/processed_ranking/train.csv" ]; then
    TRAIN_LINES=$(wc -l < "datasets/02_rtptorrent/processed_ranking/train.csv")
    log "Training samples: ${TRAIN_LINES}"
fi

if [ -f "datasets/02_rtptorrent/processed_ranking/test.csv" ]; then
    TEST_LINES=$(wc -l < "datasets/02_rtptorrent/processed_ranking/test.csv")
    log "Test samples: ${TEST_LINES}"
fi

# =============================================================================
# STEP 1: FILO-PRIORI V10 (Full Architecture)
# =============================================================================
log_header "STEP 1: Training and Evaluating Filo-Priori V10"

log "Running Filo-Priori with RTPTorrent Evaluator..."
log "  - Full evaluation against 7 pre-computed baselines"
log "  - Per-project APFD comparison"
log "  - Statistical significance tests"

START_TIME=$(date +%s)

# Set environment variable to avoid tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Run the main RTPTorrent pipeline with proper evaluator
python main_rtptorrent.py --config configs/experiment_rtptorrent_l2r.yaml 2>&1 | tee -a "${LOG_FILE}"

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
log_success "Filo-Priori L2R completed in ${TOTAL_TIME} seconds"

# =============================================================================
# STEP 2: COPY RESULTS
# =============================================================================
log_header "STEP 2: Collecting Results"

# Copy results from main_rtptorrent.py results location
FILO_RESULTS="results/experiment_rtptorrent_l2r"
if [ -d "${FILO_RESULTS}" ]; then
    cp -r "${FILO_RESULTS}"/* "${RESULTS_BASE}/" 2>/dev/null || true
    log_success "Results copied from ${FILO_RESULTS}"
else
    log_error "Filo-Priori results not found at ${FILO_RESULTS}"
fi

# =============================================================================
# STEP 3: GENERATE COMPARISON REPORT
# =============================================================================
log_header "STEP 3: Generating Comparison Report"

python3 << 'PYTHON_SCRIPT'
import json
import os
import glob
from pathlib import Path

results_base = os.environ.get('RESULTS_BASE', 'results/rtptorrent_comparison')

print("\n" + "="*70)
print("FINAL RESULTS COMPARISON - RTPTorrent Dataset (02_rtptorrent)")
print("="*70)

# Look for Filo-Priori results.json
v10_results_file = f"{results_base}/results.json"
if not os.path.exists(v10_results_file):
    v10_results_file = "results/experiment_rtptorrent_l2r/results.json"

if os.path.exists(v10_results_file):
    with open(v10_results_file, 'r') as f:
        data = json.load(f)

    print(f"\nFilo-Priori V10 Results:")
    print("-" * 70)

    # Model results (V10 format uses 'v10' key)
    v10_data = data.get('v10', data.get('model', {}))
    if v10_data:
        print(f"\n  Filo-Priori V10:")
        print(f"    Mean APFD: {v10_data.get('apfd', v10_data.get('apfd_mean', 'N/A')):.4f}")
        print(f"    Std APFD:  {v10_data.get('apfd_std', 'N/A'):.4f}")
        print(f"    NDCG@10:   {v10_data.get('ndcg_at_10', 'N/A'):.4f}")
        print(f"    Builds:    {v10_data.get('num_builds', 'N/A')}")

    # Baselines
    if 'baselines' in data:
        print(f"\n  Baselines:")
        for baseline, metrics in data['baselines'].items():
            if isinstance(metrics, dict):
                apfd = metrics.get('apfd', metrics.get('mean_apfd', 'N/A'))
                print(f"    {baseline:25s}: APFD = {apfd:.4f}")
            else:
                print(f"    {baseline:25s}: APFD = {metrics:.4f}")

    # Improvement
    if 'improvement_vs_rf' in data:
        print(f"\n  Improvement vs Recently-Failed: {data['improvement_vs_rf']:.2f}%")

    if 'p_value' in data:
        p = data['p_value']
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  p-value: {p:.6f} {sig}")

    print("\n" + "="*70)
    print("RANKING SUMMARY")
    print("="*70)

    methods = []
    v10_data = data.get('v10', data.get('model', {}))
    if v10_data:
        model_apfd = v10_data.get('apfd', v10_data.get('apfd_mean', 0))
        methods.append(('Filo-Priori V10 (Ours)', model_apfd))

    if 'baselines' in data:
        for baseline, metrics in data['baselines'].items():
            if isinstance(metrics, dict):
                apfd = metrics.get('apfd', metrics.get('mean_apfd', 0))
            else:
                apfd = metrics
            methods.append((baseline, apfd))

    # Sort by APFD
    methods.sort(key=lambda x: x[1], reverse=True)

    print("\nRanking by Mean APFD:")
    print("-" * 60)
    for i, (method, apfd) in enumerate(methods, 1):
        marker = " <-- OURS" if "Filo" in method or "V10" in method else ""
        print(f"  {i}. {method:35s} APFD: {apfd:.4f}{marker}")

    # Save comparison
    comparison = {
        'filo_priori_v10_apfd': methods[0][1] if methods and 'Filo' in methods[0][0] else None,
        'ranking': [(m, float(a)) for m, a in methods],
        'improvement_vs_recently_failed': data.get('improvement_vs_rf'),
        'p_value': data.get('p_value')
    }

    os.makedirs(results_base, exist_ok=True)
    with open(f"{results_base}/comparison_summary.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to: {results_base}/comparison_summary.json")

else:
    print("\nNo V10 results found.")
    print(f"Looking for: {v10_results_file}")
    print("Please check if run_v10_rtptorrent.py completed successfully.")

print("\n" + "="*70)
PYTHON_SCRIPT

# =============================================================================
# SUMMARY
# =============================================================================
log_header "EXPERIMENT COMPLETE"

echo ""
echo "============================================================"
echo "   EXPERIMENT SUMMARY"
echo "============================================================"
echo "Dataset: 02_rtptorrent"
echo "Total time: ${TOTAL_TIME}s"
echo ""
echo "Results saved to: ${RESULTS_BASE}"
echo "  - Filo-Priori results: ${RESULTS_BASE}/"
echo "  - Comparison: ${RESULTS_BASE}/comparison_summary.json"
echo "  - Log: ${LOG_FILE}"
echo ""
echo "Baselines compared:"
echo "  - untreated"
echo "  - random"
echo "  - recently-failed"
echo "  - optimal-failure"
echo "  - optimal-failure-duration"
echo "  - matrix-naive"
echo "  - matrix-conditional-prob"
echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"

log "Experiment finished successfully"
