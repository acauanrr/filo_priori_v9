#!/bin/bash
# Setup Script for BGE Fine-Tuning
# Installs required dependencies for contrastive fine-tuning

echo "======================================================================="
echo "SETUP: BGE Fine-Tuning for Software Engineering Domain"
echo "======================================================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install sentence-transformers and datasets
echo ""
echo "Installing sentence-transformers and datasets..."
pip install sentence-transformers datasets

# Install other requirements
echo ""
echo "Installing additional requirements..."
pip install pandas numpy scikit-learn tqdm PyYAML matplotlib seaborn

# Verify installation
echo ""
echo "======================================================================="
echo "Verifying installation..."
echo "======================================================================="

python -c "
import torch
import sentence_transformers
import pandas as pd
import numpy as np

print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'✓ sentence-transformers: {sentence_transformers.__version__}')
print(f'✓ pandas: {pd.__version__}')
print(f'✓ numpy: {np.__version__}')
"

echo ""
echo "======================================================================="
echo "Setup complete!"
echo "======================================================================="
echo ""
echo "Next steps:"
echo "  1. Test triplet generation:"
echo "     python scripts/test_triplet_generation.py"
echo ""
echo "  2. Quick fine-tuning test (30 min):"
echo "     python scripts/finetune_bge.py --config configs/finetune_bge.yaml"
echo ""
echo "  3. Monitor GPU usage:"
echo "     watch -n 1 nvidia-smi"
echo ""
