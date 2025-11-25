#!/bin/bash
# Run V8 Training with 10K Sample
# This script runs the V8 training pipeline with a 10K sample for validation

echo "======================================================================="
echo "FILO-PRIORI V8 - TRAINING RUN (10K SAMPLE)"
echo "======================================================================="
echo ""
echo "Configuration:"
echo "  - Sample size: 10,000 records"
echo "  - Device: CPU"
echo "  - Model: DualStreamModelV8"
echo "  - Semantic: BGE-Large (1024-dim)"
echo "  - Structural: Historical features (6-dim)"
echo ""
echo "Expected duration: ~30-45 minutes on CPU"
echo ""
echo "Starting training..."
echo ""

# Run training
./venv/bin/python main_v8.py \
    --config configs/experiment_v8_baseline.yaml \
    --device cpu \
    --sample-size 10000

echo ""
echo "======================================================================="
echo "Training complete!"
echo "======================================================================="
