#!/bin/bash

# Project Cleanup Script
# Executes the cleanup plan defined in CLEANUP_PLAN.md
# Run with: bash cleanup_project.sh

set -e  # Exit on error

echo "================================================================================"
echo "PROJECT CLEANUP SCRIPT"
echo "Filo-Priori V7 - Dual-Stream GNN"
echo "================================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Confirmation
echo -e "${YELLOW}This script will:${NC}"
echo "  1. Create new directories (scripts/, tests/)"
echo "  2. Move 13+ files to appropriate locations"
echo "  3. Delete 7 temporary/obsolete files"
echo "  4. Organize reports/ directory"
echo ""
echo -e "${RED}WARNING: This will modify your project structure!${NC}"
echo ""
read -p "Do you want to proceed? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "================================================================================"
echo "STEP 1: Creating new directories"
echo "================================================================================"

mkdir -p scripts
mkdir -p tests
mkdir -p experiments
mkdir -p reports/experiments
mkdir -p reports/architecture
mkdir -p reports/implementation
mkdir -p reports/planning
mkdir -p reports/archive

echo -e "${GREEN}✓${NC} Directories created"

echo ""
echo "================================================================================"
echo "STEP 2: Moving files to scripts/"
echo "================================================================================"

if [ -f "extract_all_metrics.py" ]; then
    mv extract_all_metrics.py scripts/
    echo -e "${GREEN}✓${NC} Moved extract_all_metrics.py"
fi

if [ -f "extract_metrics.sh" ]; then
    mv extract_metrics.sh scripts/
    chmod +x scripts/extract_metrics.sh
    echo -e "${GREEN}✓${NC} Moved extract_metrics.sh"
fi

if [ -f "simple_metrics.py" ]; then
    mv simple_metrics.py scripts/
    echo -e "${GREEN}✓${NC} Moved simple_metrics.py"
fi

if [ -f "verify_test_dataset.py" ]; then
    mv verify_test_dataset.py scripts/
    echo -e "${GREEN}✓${NC} Moved verify_test_dataset.py"
fi

echo ""
echo "================================================================================"
echo "STEP 3: Moving files to tests/"
echo "================================================================================"

if [ -f "test_apfd_simple.py" ]; then
    mv test_apfd_simple.py tests/test_apfd.py
    echo -e "${GREEN}✓${NC} Moved test_apfd_simple.py -> tests/test_apfd.py"
fi

if [ -f "test_config_verification.py" ]; then
    mv test_config_verification.py tests/
    echo -e "${GREEN}✓${NC} Moved test_config_verification.py"
fi

if [ -f "test_imports.py" ]; then
    mv test_imports.py tests/
    echo -e "${GREEN}✓${NC} Moved test_imports.py"
fi

if [ -f "test_safeguards_simple.py" ]; then
    mv test_safeguards_simple.py tests/test_safeguards.py
    echo -e "${GREEN}✓${NC} Moved test_safeguards_simple.py -> tests/test_safeguards.py"
fi

if [ -f "verify_architecture.py" ]; then
    mv verify_architecture.py tests/
    echo -e "${GREEN}✓${NC} Moved verify_architecture.py"
fi

if [ -f "verify_complete_pipeline.py" ]; then
    mv verify_complete_pipeline.py tests/verify_pipeline_complete.py
    echo -e "${GREEN}✓${NC} Moved verify_complete_pipeline.py -> tests/verify_pipeline_complete.py"
fi

if [ -f "verify_pipeline.py" ]; then
    mv verify_pipeline.py tests/
    chmod +x tests/verify_pipeline.py
    echo -e "${GREEN}✓${NC} Moved verify_pipeline.py"
fi

if [ -f "verify_system.py" ]; then
    mv verify_system.py tests/
    echo -e "${GREEN}✓${NC} Moved verify_system.py"
fi

echo ""
echo "================================================================================"
echo "STEP 4: Moving utils.py to src/"
echo "================================================================================"

if [ -f "utils.py" ]; then
    mv utils.py src/
    echo -e "${GREEN}✓${NC} Moved utils.py -> src/utils.py"
fi

echo ""
echo "================================================================================"
echo "STEP 5: Moving old experiment scripts to experiments/"
echo "================================================================================"

if [ -f "run_experiment_008.sh" ]; then
    mv run_experiment_008.sh experiments/
    echo -e "${GREEN}✓${NC} Moved run_experiment_008.sh"
fi

if [ -f "run_experiment_009.sh" ]; then
    mv run_experiment_009.sh experiments/
    echo -e "${GREEN}✓${NC} Moved run_experiment_009.sh"
fi

if [ -f "run_experiment_010.sh" ]; then
    mv run_experiment_010.sh experiments/
    echo -e "${GREEN}✓${NC} Moved run_experiment_010.sh"
fi

if [ -f "run_experiment_011.sh" ]; then
    mv run_experiment_011.sh experiments/
    echo -e "${GREEN}✓${NC} Moved run_experiment_011.sh"
fi

echo ""
echo "================================================================================"
echo "STEP 6: Deleting temporary/obsolete files"
echo "================================================================================"

files_to_delete=(
    "calculate_apfd_experiment_012.py"
    "recalculate_apfd_exp012.py"
    "example_usage.py"
    "run_experiment_004.sh"
    "run_experiment_006.sh"
    "test_config_004.py"
    "test_experiment_006_config.py"
)

for file in "${files_to_delete[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo -e "${GREEN}✓${NC} Deleted $file"
    else
        echo -e "${YELLOW}○${NC} $file not found (skipping)"
    fi
done

echo ""
echo "================================================================================"
echo "STEP 7: Moving documentation"
echo "================================================================================"

if [ -f "EXPERIMENT_012_SUMMARY.md" ]; then
    mv EXPERIMENT_012_SUMMARY.md reports/experiments/experiment_012_summary.md
    echo -e "${GREEN}✓${NC} Moved EXPERIMENT_012_SUMMARY.md"
fi

echo ""
echo "================================================================================"
echo "STEP 8: Organizing reports/ directory"
echo "================================================================================"

# Move experiment reports
for file in reports/EXPERIMENT_*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        mv "$file" "reports/experiments/$filename"
        echo -e "${GREEN}✓${NC} Moved $filename to experiments/"
    fi
done

# Move architecture docs
for pattern in "ARCHITECTURE*" "PIPELINE*" "SYSTEM*"; do
    for file in reports/${pattern}.md; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            mv "$file" "reports/architecture/$filename"
            echo -e "${GREEN}✓${NC} Moved $filename to architecture/"
        fi
    done
done

# Move implementation docs
for pattern in "APFD*" "IMPLEMENTATION*" "SECTION*" "FAST_NJ*"; do
    for file in reports/${pattern}.md; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            mv "$file" "reports/implementation/$filename"
            echo -e "${GREEN}✓${NC} Moved $filename to implementation/"
        fi
    done
done

# Move planning docs
for pattern in "STRATEGIC*" "EXPERIMENTAL*" "PHASE*" "PLAN*"; do
    for file in reports/${pattern}.md; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            mv "$file" "reports/planning/$filename"
            echo -e "${GREEN}✓${NC} Moved $filename to planning/"
        fi
    done
done

# Move analysis/fix docs to archive
for pattern in "ANALYSIS*" "FIX*" "ISSUE*" "CHANGES*" "CRITICAL*" "VERIFICATION*"; do
    for file in reports/${pattern}.md; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            mv "$file" "reports/archive/$filename"
            echo -e "${GREEN}✓${NC} Moved $filename to archive/"
        fi
    done
done

echo ""
echo "================================================================================"
echo "STEP 9: Creating README files for new directories"
echo "================================================================================"

# scripts/README.md
cat > scripts/README.md << 'EOF'
# Scripts

Utility scripts for analysis and data processing.

## Files

- `extract_all_metrics.py` - Extract metrics from all experiments
- `extract_metrics.sh` - Shell script to extract metrics
- `simple_metrics.py` - Simple metrics extraction
- `verify_test_dataset.py` - Verify test dataset structure

## Usage

```bash
python scripts/verify_test_dataset.py
bash scripts/extract_metrics.sh
```
EOF

echo -e "${GREEN}✓${NC} Created scripts/README.md"

# tests/README.md
cat > tests/README.md << 'EOF'
# Tests

Test and verification scripts.

## Files

- `test_apfd.py` - Test APFD calculation
- `test_imports.py` - Test imports
- `test_config_verification.py` - Verify configuration
- `test_safeguards.py` - Test safeguards
- `verify_architecture.py` - Verify architecture
- `verify_pipeline.py` - Verify pipeline
- `verify_pipeline_complete.py` - Complete pipeline verification
- `verify_system.py` - System verification

## Usage

```bash
python tests/test_imports.py
python tests/test_apfd.py
```
EOF

echo -e "${GREEN}✓${NC} Created tests/README.md"

# experiments/README.md
cat > experiments/README.md << 'EOF'
# Experiments (Phase II)

Experiment scripts for Phase II architectural refinements.

## Files

- `run_experiment_008.sh` - GATv2 architecture
- `run_experiment_009.sh` - Attention pooling
- `run_experiment_010.sh` - Bidirectional fusion
- `run_experiment_011.sh` - Improved classifier

## Usage

```bash
bash experiments/run_experiment_008.sh
```

**Note:** These are Phase II experiments. For current best practices, use:
```bash
bash run_experiment_012.sh  # In project root
```
EOF

echo -e "${GREEN}✓${NC} Created experiments/README.md"

# reports/README.md
cat > reports/README.md << 'EOF'
# Reports and Documentation

Organized documentation for the Filo-Priori V7 project.

## Directory Structure

- **experiments/** - Experiment evaluation reports and summaries
- **architecture/** - Architecture documentation and analysis
- **implementation/** - Implementation guides and summaries
- **planning/** - Strategic plans and experimental protocols
- **archive/** - Historical analysis and fix documentation

## Key Documents

### Experiments
- `experiments/experiment_007_evaluation.md` - Phase I baseline evaluation
- `experiments/experiment_012_summary.md` - Best practices summary

### Implementation
- `implementation/APFD_INTEGRATION_GUIDE.md` - Complete APFD integration guide
- `implementation/APFD_IMPLEMENTATION_SUMMARY.md` - APFD summary

### Planning
- `planning/STRATEGIC_RECOVERY_PLAN.md` - Overall strategy
- `planning/EXPERIMENTAL_PROTOCOL.md` - Experimental protocol

## Navigation

For quick access to specific topics:
- APFD/Prioritization: `implementation/APFD_*.md`
- Experiment results: `experiments/`
- Architecture details: `architecture/`
EOF

echo -e "${GREEN}✓${NC} Created reports/README.md"

echo ""
echo "================================================================================"
echo "CLEANUP COMPLETED SUCCESSFULLY!"
echo "================================================================================"
echo ""
echo -e "${GREEN}✓${NC} Project structure cleaned and organized"
echo ""
echo "Summary:"
echo "  - Created: scripts/, tests/, experiments/"
echo "  - Moved: 13+ files to appropriate locations"
echo "  - Deleted: 7 temporary/obsolete files"
echo "  - Organized: reports/ into subdirectories"
echo ""
echo "Next steps:"
echo "  1. Review the new structure"
echo "  2. Update imports in moved files (if necessary)"
echo "  3. Test that main.py still works"
echo "  4. Commit changes to git"
echo ""
echo "Files remaining in root:"
ls -1 | grep -E '\.(py|sh|md)$' | head -20
echo ""
echo "================================================================================"
