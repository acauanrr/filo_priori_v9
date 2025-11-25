"""
Simple V8 Validation Script

Tests the core V8 components without complex imports.

Usage:
    python test_v8_simple.py
"""

import sys
import os
import torch
import pandas as pd
import numpy as np

print("="*70)
print("SIMPLE V8 VALIDATION TEST")
print("="*70)

# Test 1: Structural Features
print("\n1. Testing structural feature extraction...")
sys.path.insert(0, 'src')
from preprocessing.structural_feature_extractor import StructuralFeatureExtractor

df_train = pd.read_csv('datasets/train.csv').head(200)
extractor = StructuralFeatureExtractor(recent_window=5)
extractor.fit(df_train)
features = extractor.transform(df_train, is_test=False)

print(f"   ✓ Features extracted: {features.shape}")
print(f"   ✓ Feature range: [{features.min():.2f}, {features.max():.2f}]")

# Test 2: Phylogenetic Graph
print("\n2. Testing phylogenetic graph builder...")
from phylogenetic.phylogenetic_graph_builder import PhylogeneticGraphBuilder

builder = PhylogeneticGraphBuilder(graph_type='co_failure')
builder.fit(df_train)
stats = builder.get_graph_statistics()

print(f"   ✓ Graph type: {stats['graph_type']}")
print(f"   ✓ Nodes: {stats['num_nodes']}")
print(f"   ✓ Edges: {stats['num_edges']}")

# Test 3: Model Architecture
print("\n3. Testing V8 model architecture...")

# Load dual_stream_v8.py directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "dual_stream_v8",
    "src/models/dual_stream_v8.py"
)
dual_stream_v8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dual_stream_v8)

# Create model
model = dual_stream_v8.DualStreamModelV8(
    semantic_config={'input_dim': 1024, 'hidden_dim': 256},
    structural_config={'input_dim': 6, 'hidden_dim': 256}
)

print(f"   ✓ Model created")

# Test forward pass
batch_size = 8
semantic_input = torch.randn(batch_size, 1024)
structural_input = torch.randn(batch_size, 6)

with torch.no_grad():
    logits = model(semantic_input=semantic_input, structural_input=structural_input)

print(f"   ✓ Forward pass successful")
print(f"   ✓ Input shapes: semantic=[{batch_size}, 1024], structural=[{batch_size}, 6]")
print(f"   ✓ Output shape: {logits.shape}")

assert logits.shape == (batch_size, 2), f"Expected [{batch_size}, 2], got {logits.shape}"

# Test 4: Feature representations
print("\n4. Testing feature extraction...")
with torch.no_grad():
    sem_feat, struct_feat, fused = model.get_feature_representations(
        semantic_input, structural_input
    )

print(f"   ✓ Semantic features: {sem_feat.shape}")
print(f"   ✓ Structural features: {struct_feat.shape}")
print(f"   ✓ Fused features: {fused.shape}")

# Test 5: Integration test
print("\n5. Testing end-to-end integration...")
from torch.utils.data import TensorDataset, DataLoader

# Create dummy data
embeddings = np.random.randn(100, 1024).astype(np.float32)
struct_features = features[:100]
labels = np.random.randint(0, 2, size=100)

dataset = TensorDataset(
    torch.FloatTensor(embeddings),
    torch.FloatTensor(struct_features),
    torch.LongTensor(labels)
)

loader = DataLoader(dataset, batch_size=16, shuffle=False)

model.eval()
all_logits = []

with torch.no_grad():
    for emb, struct, lbl in loader:
        logits = model(semantic_input=emb, structural_input=struct)
        all_logits.append(logits)

all_logits = torch.cat(all_logits, dim=0)
probs = torch.softmax(all_logits, dim=1)
preds = torch.argmax(all_logits, dim=1)

print(f"   ✓ Processed {len(dataset)} samples in {len(loader)} batches")
print(f"   ✓ Predictions: {preds.shape}")
print(f"   ✓ Probabilities: {probs.shape}")

# Summary
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)
print("✓ TEST 1: Structural feature extraction - PASSED")
print("✓ TEST 2: Phylogenetic graph construction - PASSED")
print("✓ TEST 3: V8 model architecture - PASSED")
print("✓ TEST 4: Feature extraction - PASSED")
print("✓ TEST 5: End-to-end integration - PASSED")
print("="*70)
print("\n✅ ALL TESTS PASSED!")
print("\nV8 pipeline is ready for training!")
print("\nNext steps:")
print("  1. Install full dependencies: pip install -r requirements.txt")
print("  2. Run training: python main_v8.py --config configs/experiment_v8_baseline.yaml")
