#!/usr/bin/env bash
set -euo pipefail

# Experiment 017: PIPELINE CORRECTION - Fix fundamental issues from Exp 016_01
#
# KEY CORRECTIONS:
# 1. Group-aware data split (GroupShuffleSplit by Build_ID) - prevents leakage
# 2. GroupedBatchSampler - batches contain samples from same build(s) for ranking
# 3. Curriculum learning - ranking loss starts after classification is learned
# 4. Improved monitoring - AUPRC macro + probability calibration
#
# Expected Impact:
# - Better generalization to unseen builds (no leakage)
# - Effective ranking loss (proper pairs within batches)
# - Smoother training (no conflicting objectives early on)
# - Better calibrated probabilities for APFD

CONFIG="configs/experiment_017_ranking_corrected.yaml"
REWIRE_CFG="configs/rewiring_017.yaml"  # Updated config with k=20
OUT_DIR="results/experiment_017_ranking_corrected"
GRAPH_PKL="$OUT_DIR/trees/graph_structure.pkl"
GRAPH_DATA_PT="$OUT_DIR/graph_data.pt"
REWIRED_PT="$OUT_DIR/rewiring/rewired_graph.pt"

echo "[017] =========================================="
echo "[017] EXPERIMENT 017: Pipeline Correction"
echo "[017] =========================================="
echo "[017] Fixing fundamental issues from Exp 016_01:"
echo "[017]   ✓ Group-aware split (anti-leakage)"
echo "[017]   ✓ Grouped batch sampler (pro-ranking)"
echo "[017]   ✓ Curriculum learning (smooth convergence)"
echo "[017]   ✓ Improved monitoring (AUPRC + calibration)"
echo "[017] =========================================="
echo ""

echo "[017] Step 0: Create output directories"
mkdir -p "$OUT_DIR" "$OUT_DIR/rewiring" "$OUT_DIR/embeddings" "$OUT_DIR/trees"

echo "[017] Step 1: Ensuring torch-scatter is installed"
python - <<'PY'
import sys
try:
    import torch_scatter  # noqa: F401
    print("✓ torch-scatter already installed")
except Exception:
    print("Installing torch-scatter for PyTorch 2.9.0+cu128...")
    import subprocess, sys, torch

    ver_parts = torch.__version__.split('+')[0].split('.')
    ver = f"{ver_parts[0]}.{ver_parts[1]}.0"  # 2.9.0
    cuda = torch.version.cuda  # "12.8"

    if cuda:
        cu = cuda.replace('.', '')  # "128"
        index = f"https://data.pyg.org/whl/torch-{ver}+cu{cu}.html"
    else:
        index = f"https://data.pyg.org/whl/torch-{ver}+cpu.html"

    # Try PyG wheel index
    cmd = [sys.executable, '-m', 'pip', 'install', 'torch-scatter', '-f', index]
    print(f"Installing: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print("✓ torch-scatter installed from PyG wheels")
    except subprocess.CalledProcessError:
        # Fallback: Install with --no-build-isolation
        print("PyG wheels not available, trying with --no-build-isolation...")
        cmd = [sys.executable, '-m', 'pip', 'install', 'torch-scatter', '--no-build-isolation']
        subprocess.check_call(cmd)
        print("✓ torch-scatter compiled successfully")
print("✓ torch-scatter ready")
PY

echo ""
echo "[017] Step 2: First pass – build embeddings + initial k-NN graph"
echo "[017] NOTE: Using GROUP-AWARE split (GroupShuffleSplit by Build_ID)"
echo "[017]       This prevents leakage and ensures proper generalization"
echo ""
# First run will fallback to k-NN and cache graph_structure.pkl
python main.py --config "$CONFIG" || true

echo ""
echo "[017] Step 3: Exporting graph_data.pt for rewiring"
python - <<PY
import os, pickle, torch, numpy as np, yaml
out_dir = os.path.abspath("$OUT_DIR")
graph_pkl = os.path.abspath("$GRAPH_PKL")
graph_pt = os.path.abspath("$GRAPH_DATA_PT")
config_path = os.path.abspath("$CONFIG")

# Load config to check if multi-field is enabled
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

use_multi_field = config.get('embedding', {}).get('use_multi_field', False)

# Load embeddings (different paths for single vs multi-field)
if use_multi_field:
    print("Loading multi-field embeddings...")
    emb_dir = os.path.join(out_dir, "embeddings", "train")

    # Check if field embeddings exist
    field_files = ['summary_embeddings.npy', 'steps_embeddings.npy',
                   'commits_embeddings.npy', 'CR_embeddings.npy']

    for fname in field_files:
        fpath = os.path.join(emb_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Field embedding not found: {fpath}")

    # Load field embeddings
    field_embeddings = [np.load(os.path.join(emb_dir, fname)) for fname in field_files]

    # Fuse embeddings using field fusion module
    from src.embeddings.field_fusion import create_field_fusion

    # Convert to tensors
    field_tensors = [torch.tensor(emb, dtype=torch.float32) for emb in field_embeddings]

    # Create fusion module and fuse
    fusion_module = create_field_fusion(config['embedding'])
    fused_embeddings = fusion_module(field_tensors)
    emb = fused_embeddings.detach().cpu().numpy()

    print(f"Fused embeddings shape: {emb.shape}")
else:
    print("Loading single-field embeddings...")
    emb_npy = os.path.join(out_dir, "embeddings", "train_embeddings.npy")

    if not os.path.exists(emb_npy):
        raise FileNotFoundError(f"Train embeddings not found: {emb_npy}")

    emb = np.load(emb_npy)
    print(f"Embeddings shape: {emb.shape}")

# Load graph structure
if not os.path.exists(graph_pkl):
    raise FileNotFoundError(f"Graph cache not found: {graph_pkl}")

with open(graph_pkl, 'rb') as f:
    g = pickle.load(f)

edge_index = torch.tensor(g['edge_index'], dtype=torch.long)
edge_weights = torch.tensor(g['edge_weights'], dtype=torch.float32)
node_features = torch.tensor(emb, dtype=torch.float32)

os.makedirs(os.path.dirname(graph_pt), exist_ok=True)
torch.save({
    'node_features': node_features,
    'edge_index': edge_index,
    'edge_weights': edge_weights,
}, graph_pt)
print(f"✓ Exported graph_data.pt to {graph_pt}")
print(f"  Node features: {node_features.shape}")
print(f"  Edge index: {edge_index.shape}")
print(f"  Edge weights: {edge_weights.shape}")
PY

echo ""
echo "[017] Step 4: Validating rewiring config"
echo "[017] Config: configs/rewiring_017.yaml (k=20, keep_ratio=0.2)"
echo ""

# Verificar se config existe
if [ ! -f "configs/rewiring_017.yaml" ]; then
  echo "❌ ERROR: Rewiring config not found at configs/rewiring_017.yaml"
  echo "   Please ensure the config file exists before running this script."
  exit 1
fi

# Verificar se graph_data.pt existe antes de rewiring
if [ ! -f "$GRAPH_DATA_PT" ]; then
  echo "❌ ERROR: Graph data not found at $GRAPH_DATA_PT"
  echo "   The graph data should have been created in Step 3."
  exit 1
fi

echo "✓ Rewiring config validated (using k=20, scoring_device=cpu)"
echo "✓ Graph data found at $GRAPH_DATA_PT"

echo ""
echo "[017] Step 5: Running graph rewiring (k=20, keep_ratio=0.2)"
python run_graph_rewiring.py --config "configs/rewiring_017.yaml"

if [ ! -f "$REWIRED_PT" ]; then
  echo "ERROR: Rewired graph not found at $REWIRED_PT"
  exit 1
fi

echo ""
echo "[017] =========================================="
echo "[017] Step 6: FINAL TRAINING with ALL CORRECTIONS"
echo "[017] =========================================="
echo "[017] Training features:"
echo "[017]   ✓ Group-aware split (no build leakage)"
echo "[017]   ✓ GroupedBatchSampler (ranking-aware batches)"
echo "[017]   ✓ Curriculum learning (ranking starts epoch 5, ramps to epoch 10)"
echo "[017]   ✓ Reduced ranking weight (0.15 instead of 0.5)"
echo "[017]   ✓ Score type: prob (aligns with APFD)"
echo "[017]   ✓ Loss type: logistic (smooth, no margin conflicts)"
echo "[017]   ✓ Monitoring: AUPRC macro (robust to imbalance)"
echo "[017]   ✓ Probability calibration: enabled"
echo "[017] =========================================="
echo ""

python main.py --config "$CONFIG"

echo ""
echo "[017] =========================================="
echo "[017] EXPERIMENT 017 COMPLETE!"
echo "[017] =========================================="
echo "[017] Results saved to: $OUT_DIR"
echo ""
echo "[017] Expected improvements over Exp 016_01:"
echo "[017]   • Better generalization (no leakage)"
echo "[017]   • Higher APFD (effective ranking)"
echo "[017]   • Smoother convergence (curriculum)"
echo "[017]   • Better calibration (temperature scaling)"
echo "[017] =========================================="
