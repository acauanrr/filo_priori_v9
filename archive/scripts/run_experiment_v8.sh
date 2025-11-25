#!/bin/bash
# Run V8 Baseline Experiment with Fine-tuned BGE Model
# This script clears caches and runs the full training pipeline

echo "======================================================================="
echo "FILO-PRIORI V8 - BASELINE EXPERIMENT"
echo "======================================================================="
echo ""
echo "Configuration:"
echo "  - Model: Fine-tuned BGE (models/finetuned_bge_v1)"
echo "  - Architecture: DualStreamModelV8 with GAT"
echo "  - Config: configs/experiment_v8_baseline.yaml"
echo ""

# Ask for confirmation to clear caches
read -p "Clear caches before running? (recommended) [Y/n]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Clearing caches..."
    rm -rf cache/embeddings/*.npy
    rm -f cache/structural_features.pkl
    rm -f cache/phylogenetic_graph.pkl
    echo "✓ Caches cleared"
fi

echo ""
echo "Starting training..."
echo "======================================================================="
echo ""

# Detect Python executable
if [ -f "/home/acauanribeiro/iats/filo_priori_v8/venv/bin/python" ]; then
    PYTHON="/home/acauanribeiro/iats/filo_priori_v8/venv/bin/python"
elif [ -f "./venv/bin/python" ]; then
    PYTHON="./venv/bin/python"
else
    PYTHON="python3"
fi

echo "Using Python: $PYTHON"
echo ""

# Detect device (cuda or cpu)
DEVICE="cuda"
read -p "Use GPU (cuda) or CPU? [cuda/cpu, default: cuda]: " device_input
if [ -n "$device_input" ]; then
    DEVICE="$device_input"
fi

echo "Using device: $DEVICE"
echo ""

# Run training
$PYTHON main_v8.py \
    --config configs/experiment_v8_baseline.yaml \
    --device $DEVICE

echo ""
echo "======================================================================="
echo "Training complete!"
echo "Results saved to: results/experiment_v8_baseline/"
echo "======================================================================="
