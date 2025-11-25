#!/usr/bin/env bash
set -euo pipefail

echo "=== Setting up Python environment for Experiment 015 ==="
echo ""

# Ensure venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

echo "Step 1: Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Step 2: Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "Step 3: Installing requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Step 4: Installing torch-scatter..."
# Get actual torch version
TORCH_VERSION=$(python -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(f'{v[0]}.{v[1]}.0')")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.','')[:3] if torch.cuda.is_available() else 'cpu')")
echo "Detected: PyTorch ${TORCH_VERSION}, CUDA ${CUDA_VERSION}"

pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html

echo ""
echo "Step 5: Verifying torch-scatter installation..."
python -c "import torch_scatter; print('✓ torch-scatter installed successfully')"

echo ""
echo "=== Environment setup complete! ==="
echo "You can now run: bash run_experiment_015.sh"
