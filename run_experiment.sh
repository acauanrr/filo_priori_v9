#!/bin/bash
#
# Filo-Priori Experiment Runner
#
# Simplified execution script with intelligent embedding caching
#
# Usage:
#   ./run_experiment.sh                    # Run with cache (fast)
#   ./run_experiment.sh --force-regen      # Force regenerate embeddings
#   ./run_experiment.sh --clear-cache      # Clear cache and regenerate
#   ./run_experiment.sh --help             # Show help

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
CACHE_DIR="$PROJECT_DIR/cache"
RESULTS_DIR="$PROJECT_DIR/results"
LOGS_DIR="$PROJECT_DIR/logs"

# Default arguments
FORCE_REGEN=false
CLEAR_CACHE=false
CONFIG_FILE="configs/experiment.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force-regen|--force)
            FORCE_REGEN=true
            shift
            ;;
        --clear-cache|--clear)
            CLEAR_CACHE=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Filo-Priori Experiment Runner"
            echo ""
            echo "Usage:"
            echo "  ./run_experiment.sh [options]"
            echo ""
            echo "Options:"
            echo "  --force-regen, --force    Force regenerate embeddings"
            echo "  --clear-cache, --clear    Clear cache before running"
            echo "  --config FILE             Use custom config file"
            echo "  --help, -h                Show this help"
            echo ""
            echo "Examples:"
            echo "  ./run_experiment.sh                    # Run with cache (recommended)"
            echo "  ./run_experiment.sh --force-regen      # Regenerate embeddings"
            echo "  ./run_experiment.sh --clear-cache      # Fresh start"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Header
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}               FILO-PRIORI EXPERIMENT RUNNER${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Check Python virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Check/install dependencies
echo -e "${BLUE}Checking dependencies...${NC}"
if ! python -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install --upgrade pip -q
    pip install torch sentence-transformers pandas pyyaml numpy tqdm scikit-learn matplotlib seaborn torch-geometric -q
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Dependencies already installed${NC}"
fi
echo ""

# Clear cache if requested
if [ "$CLEAR_CACHE" = true ]; then
    echo -e "${YELLOW}Clearing cache...${NC}"
    rm -rf "$CACHE_DIR"
    mkdir -p "$CACHE_DIR"
    echo -e "${GREEN}✓ Cache cleared${NC}"
    echo ""
fi

# Create directories
mkdir -p "$CACHE_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

# Show cache status
echo -e "${BLUE}Cache Status:${NC}"
if [ -f "$CACHE_DIR/embeddings.npz" ]; then
    CACHE_SIZE=$(du -h "$CACHE_DIR/embeddings.npz" | cut -f1)
    echo -e "  ${GREEN}✓ Embeddings cached${NC} (${CACHE_SIZE})"

    if [ "$FORCE_REGEN" = true ]; then
        echo -e "  ${YELLOW}⚠ Force regeneration enabled${NC}"
    else
        echo -e "  ${GREEN}→ Will reuse cached embeddings${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠ No cache found${NC}"
    echo -e "  ${YELLOW}→ Will generate embeddings (may take 1-2 minutes)${NC}"
fi
echo ""

# Show config
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Config file: ${CONFIG_FILE}"
echo -e "  Force regen: ${FORCE_REGEN}"
echo ""

# Run experiment
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}                    STARTING EXPERIMENT${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Build command
CMD="python main.py --config $CONFIG_FILE"
if [ "$FORCE_REGEN" = true ]; then
    CMD="$CMD --force-regen-embeddings"
fi

# Run
echo -e "${GREEN}Executing: $CMD${NC}"
echo ""

$CMD
