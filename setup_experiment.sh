#!/bin/bash
################################################################################
# Filo-Priori Experiment Setup Script
#
# This script sets up the environment for running experiments.
# Run this once before executing experiments.
#
# Usage:
#   ./setup_experiment.sh
#
# Author: Filo-Priori Team
# Date: 2025-11-10
################################################################################

set -e  # Exit on error

echo "=============================================================================="
echo "Filo-Priori Experiment Setup"
echo "=============================================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check if running from project root
if [ ! -f "main.py" ]; then
    print_error "Error: main.py not found. Please run this script from the project root."
    exit 1
fi

print_info "Detected project root: $(pwd)"
echo ""

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
    print_success "Python $PYTHON_VERSION detected"
else
    print_error "Python $REQUIRED_VERSION or higher required. Found: $PYTHON_VERSION"
    exit 1
fi
echo ""

# Step 2: Create/activate virtual environment
echo "Step 2: Virtual environment setup..."
USE_VENV=true

if [ -d "venv" ]; then
    print_info "Virtual environment already exists at venv/"
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing old venv..."
        rm -rf venv
        print_info "Creating new virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created at venv/"
    else
        print_success "Using existing venv"
    fi
else
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created at venv/"
fi

# Activate venv
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"
echo ""

# Step 3: Install/update dependencies
echo "Step 3: Installing Python dependencies..."
print_info "Installing all dependencies from requirements.txt..."
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

print_success "All dependencies installed"
echo ""

# Step 4: Verify critical packages
echo "Step 4: Verifying critical packages..."

# Check PyTorch
python -c "import torch; print('PyTorch:', torch.__version__)" 2>/dev/null && print_success "PyTorch installed" || print_error "PyTorch missing"

# Check PyTorch Geometric
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)" 2>/dev/null && print_success "PyTorch Geometric installed" || print_error "PyTorch Geometric missing"

# Check sentence-transformers
python -c "import sentence_transformers; print('sentence-transformers:', sentence_transformers.__version__)" 2>/dev/null && print_success "sentence-transformers installed" || print_error "sentence-transformers missing"

# Check transformers
python -c "import transformers; print('transformers:', transformers.__version__)" 2>/dev/null && print_success "transformers installed" || print_error "transformers missing"

echo ""

# Step 5: Check CUDA availability
echo "Step 5: Checking CUDA availability..."
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [ "$CUDA_AVAILABLE" = "True" ]; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)

    print_success "CUDA available"
    print_info "CUDA version: $CUDA_VERSION"
    print_info "GPU count: $GPU_COUNT"
    print_info "GPU 0: $GPU_NAME"
else
    print_error "CUDA not available - will run on CPU (slower)"
fi
echo ""

# Step 6: Create necessary directories
echo "Step 6: Creating necessary directories..."
mkdir -p cache/embeddings_qodo
mkdir -p logs
mkdir -p results
mkdir -p models

print_success "Directories created"
echo ""

# Step 7: Check datasets
echo "Step 7: Checking datasets..."
if [ -f "datasets/train.csv" ]; then
    TRAIN_SIZE=$(wc -l < datasets/train.csv)
    print_success "train.csv found ($TRAIN_SIZE lines)"
else
    print_error "datasets/train.csv not found"
fi

if [ -f "datasets/test.csv" ]; then
    TEST_SIZE=$(wc -l < datasets/test.csv)
    print_success "test.csv found ($TEST_SIZE lines)"
else
    print_error "datasets/test.csv not found"
fi
echo ""

# Step 8: Verify config file
echo "Step 8: Verifying configuration..."
if [ -f "configs/experiment.yaml" ]; then
    print_success "configs/experiment.yaml found"
else
    print_error "configs/experiment.yaml not found"
    print_info "Please create a config file or copy from archive/configs/"
fi
echo ""

# Step 9: Summary
echo "=============================================================================="
echo "Setup Summary"
echo "=============================================================================="
echo ""
echo "Python: $PYTHON_VERSION"
echo "CUDA: $CUDA_AVAILABLE"
if [ "$CUDA_AVAILABLE" = "True" ]; then
    echo "GPU: $GPU_NAME"
fi
echo ""
print_success "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Review config: configs/experiment.yaml"
echo "  2. Run experiment: ./run_experiment.sh"
echo ""
echo "=============================================================================="
