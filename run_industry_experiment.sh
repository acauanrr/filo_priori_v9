#!/bin/bash
# =============================================================================
# Filo-Priori vs DeepOrder - Industry Dataset (01_industry) Experiment
# =============================================================================
# Este script executa o experimento completo no dataset 01_industry:
# 1. Treina e avalia o modelo Filo-Priori
# 2. Treina e avalia o baseline DeepOrder
# 3. Compara os resultados APFD
#
# Uso:
#   ./run_industry_experiment.sh
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="configs/experiment_industry.yaml"
RESULTS_BASE="results/industry_comparison_$(date +%Y%m%d_%H%M%S)"
FILOPRIORI_RESULTS="${RESULTS_BASE}/filo_priori"
DEEPORDER_RESULTS="${RESULTS_BASE}/deeporder"
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
echo "   FILO-PRIORI vs DEEPORDER - Industry Dataset Experiment"
echo "============================================================"
echo "Config: ${CONFIG_FILE}"
echo "Results: ${RESULTS_BASE}"
echo "Started: $(date)"
echo "============================================================"
echo ""

log "Experiment started"
log "Config: ${CONFIG_FILE}"
log "Results directory: ${RESULTS_BASE}"

# =============================================================================
# STEP 1: FILO-PRIORI
# =============================================================================
log_header "STEP 1: Training and Evaluating Filo-Priori"

log "Running Filo-Priori (main.py)..."
START_TIME=$(date +%s)

# Update config to use the correct results directory
# We'll use the default config and let main.py handle the output

python main.py --config "${CONFIG_FILE}" 2>&1 | tee -a "${LOG_FILE}"

END_TIME=$(date +%s)
FILOPRIORI_TIME=$((END_TIME - START_TIME))
log_success "Filo-Priori completed in ${FILOPRIORI_TIME} seconds"

# Copy Filo-Priori results
if [ -d "results/experiment_industry" ]; then
    cp -r results/experiment_industry/* "${FILOPRIORI_RESULTS}/" 2>/dev/null || mkdir -p "${FILOPRIORI_RESULTS}"
    log "Filo-Priori results copied to ${FILOPRIORI_RESULTS}"
fi

# =============================================================================
# STEP 2: DEEPORDER BASELINE
# =============================================================================
log_header "STEP 2: Training and Evaluating DeepOrder Baseline"

log "Running DeepOrder baseline..."
START_TIME=$(date +%s)

python run_deeporder_baseline.py \
    --config "${CONFIG_FILE}" \
    --results-dir "${DEEPORDER_RESULTS}" 2>&1 | tee -a "${LOG_FILE}"

END_TIME=$(date +%s)
DEEPORDER_TIME=$((END_TIME - START_TIME))
log_success "DeepOrder completed in ${DEEPORDER_TIME} seconds"

# =============================================================================
# STEP 3: COMPARISON
# =============================================================================
log_header "STEP 3: Results Comparison"

# Extract APFD values and create comparison
python3 << 'PYTHON_SCRIPT'
import json
import os
import sys

results_base = os.environ.get('RESULTS_BASE', 'results/industry_comparison')

# Load Filo-Priori results
filo_apfd = None
filo_results_file = f"{results_base}/filo_priori/apfd_report.json"
if not os.path.exists(filo_results_file):
    # Try default location
    filo_results_file = "results/experiment_industry/apfd_report.json"

if os.path.exists(filo_results_file):
    with open(filo_results_file, 'r') as f:
        filo_data = json.load(f)
        filo_apfd = filo_data.get('mean_apfd', filo_data.get('summary', {}).get('mean_apfd'))
        if filo_apfd is None and 'statistics' in filo_data:
            filo_apfd = filo_data['statistics'].get('mean_apfd')

# Load DeepOrder results
deeporder_apfd = None
deeporder_results_file = f"{results_base}/deeporder/deeporder_results.json"
if os.path.exists(deeporder_results_file):
    with open(deeporder_results_file, 'r') as f:
        deeporder_data = json.load(f)
        deeporder_apfd = deeporder_data.get('metrics', {}).get('mean_apfd')

print("\n" + "="*60)
print("FINAL RESULTS COMPARISON - Industry Dataset (01_industry)")
print("="*60)

if filo_apfd is not None:
    print(f"\nFilo-Priori Mean APFD:  {filo_apfd:.4f}")
else:
    print("\nFilo-Priori: Results not found")

if deeporder_apfd is not None:
    print(f"DeepOrder Mean APFD:    {deeporder_apfd:.4f}")
else:
    print("DeepOrder: Results not found")

if filo_apfd is not None and deeporder_apfd is not None:
    improvement = ((filo_apfd - deeporder_apfd) / deeporder_apfd) * 100
    winner = "Filo-Priori" if filo_apfd > deeporder_apfd else "DeepOrder"
    diff = abs(filo_apfd - deeporder_apfd)

    print(f"\nDifference: {diff:.4f} ({abs(improvement):.2f}%)")
    print(f"Winner: {winner}")

    # Save comparison
    comparison = {
        'filo_priori_apfd': filo_apfd,
        'deeporder_apfd': deeporder_apfd,
        'difference': diff,
        'improvement_percent': improvement,
        'winner': winner
    }

    with open(f"{results_base}/comparison_summary.json", 'w') as f:
        json.dump(comparison, f, indent=2)

print("="*60 + "\n")
PYTHON_SCRIPT

# =============================================================================
# SUMMARY
# =============================================================================
log_header "EXPERIMENT COMPLETE"

echo ""
echo "============================================================"
echo "   EXPERIMENT SUMMARY"
echo "============================================================"
echo "Dataset: 01_industry"
echo "Filo-Priori time: ${FILOPRIORI_TIME}s"
echo "DeepOrder time: ${DEEPORDER_TIME}s"
echo "Total time: $((FILOPRIORI_TIME + DEEPORDER_TIME))s"
echo ""
echo "Results saved to: ${RESULTS_BASE}"
echo "  - Filo-Priori: ${FILOPRIORI_RESULTS}"
echo "  - DeepOrder: ${DEEPORDER_RESULTS}"
echo "  - Comparison: ${RESULTS_BASE}/comparison_summary.json"
echo "  - Log: ${LOG_FILE}"
echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"

log "Experiment finished successfully"
