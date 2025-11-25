#!/bin/bash
#
# Apply Threshold Optimization to Experiment 04a
#
# This script applies threshold optimization to the trained model from
# Experiment 04a to improve minority class recall.
#
# Expected improvements:
#   - Threshold: 0.5 → 0.05-0.15
#   - Recall Not-Pass: 0.05 → 0.25-0.35 (5-7x improvement)
#   - F1 Macro: 0.53 → 0.55-0.60
#

set -e  # Exit on error

echo "========================================================================"
echo "THRESHOLD OPTIMIZATION - EXPERIMENT 04a"
echo "========================================================================"
echo ""

# Configuration
CONFIG="configs/experiment_04a_weighted_ce_only.yaml"
MODEL="best_model_v8.pt"
OUTPUT="results/experiment_04a_weighted_ce_only"
STRATEGY="f1_macro"

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model file not found: $MODEL"
    echo "Please ensure the model from Experiment 04a is available."
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

echo "Configuration:"
echo "  Config file: $CONFIG"
echo "  Model file: $MODEL"
echo "  Output dir: $OUTPUT"
echo "  Strategy: $STRATEGY"
echo ""

echo "Starting threshold optimization..."
echo ""

# Run threshold optimization
./venv/bin/python apply_threshold_optimization.py \
    --config "$CONFIG" \
    --model-path "$MODEL" \
    --strategy "$STRATEGY" \
    --output-dir "$OUTPUT"

echo ""
echo "========================================================================"
echo "THRESHOLD OPTIMIZATION COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT/"
echo "  - threshold_optimization_results.txt"
echo "  - threshold_optimization_curves.png"
echo ""
