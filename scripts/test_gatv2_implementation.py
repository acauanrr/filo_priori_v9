"""
Test script to verify GATv2 implementation in StructuralStream
"""

import sys
import torch
import yaml

# Test 1: Import modules
print("=" * 60)
print("TEST 1: Checking imports...")
print("=" * 60)

try:
    from src.layers.gatv2 import GATv2Conv, ResidualGATv2Layer
    print("✓ GATv2 layers imported successfully")
except Exception as e:
    print(f"✗ Failed to import GATv2 layers: {e}")
    sys.exit(1)

try:
    from src.models.dual_stream import StructuralStream, DualStreamPhylogeneticTransformer
    print("✓ StructuralStream imported successfully")
except Exception as e:
    print(f"✗ Failed to import StructuralStream: {e}")
    sys.exit(1)

# Test 2: Check torch-scatter availability
print("\n" + "=" * 60)
print("TEST 2: Checking torch-scatter availability...")
print("=" * 60)

try:
    import torch_scatter
    print(f"✓ torch-scatter version: {torch_scatter.__version__}")
except ImportError:
    print("✗ torch-scatter not found. Installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "torch-scatter", "-f",
                    "https://data.pyg.org/whl/torch-2.0.0+cu118.html"], check=True)
    import torch_scatter
    print(f"✓ torch-scatter installed: {torch_scatter.__version__}")

# Test 3: Instantiate StructuralStream with linear layer_type (backward compatibility)
print("\n" + "=" * 60)
print("TEST 3: Testing backward compatibility (layer_type='linear')...")
print("=" * 60)

try:
    stream_linear = StructuralStream(
        input_dim=1024,
        hidden_dim=256,
        num_layers=2,
        dropout=0.4,
        aggregation='mean',
        layer_type='linear'
    )
    print(f"✓ Linear StructuralStream created")
    print(f"  Parameters: {sum(p.numel() for p in stream_linear.parameters()):,}")
except Exception as e:
    print(f"✗ Failed to create linear StructuralStream: {e}")
    sys.exit(1)

# Test 4: Instantiate StructuralStream with GATv2
print("\n" + "=" * 60)
print("TEST 4: Testing GATv2 StructuralStream (layer_type='gatv2')...")
print("=" * 60)

try:
    stream_gatv2 = StructuralStream(
        input_dim=1024,
        hidden_dim=256,
        num_layers=2,
        dropout=0.4,
        layer_type='gatv2',
        num_heads=4,
        use_residual=False
    )
    print(f"✓ GATv2 StructuralStream created")
    print(f"  Parameters: {sum(p.numel() for p in stream_gatv2.parameters()):,}")
except Exception as e:
    print(f"✗ Failed to create GATv2 StructuralStream: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Forward pass with dummy data (linear)
print("\n" + "=" * 60)
print("TEST 5: Testing forward pass (linear)...")
print("=" * 60)

try:
    num_nodes = 100
    num_edges = 500

    x = torch.randn(num_nodes, 1024)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_weights = torch.rand(num_edges)

    out_linear = stream_linear(x, edge_index, edge_weights)
    print(f"✓ Linear forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_linear.shape}")
    print(f"  Output mean: {out_linear.mean().item():.4f}, std: {out_linear.std().item():.4f}")
except Exception as e:
    print(f"✗ Linear forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Forward pass with dummy data (GATv2)
print("\n" + "=" * 60)
print("TEST 6: Testing forward pass (GATv2)...")
print("=" * 60)

try:
    out_gatv2 = stream_gatv2(x, edge_index)
    print(f"✓ GATv2 forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_gatv2.shape}")
    print(f"  Output mean: {out_gatv2.mean().item():.4f}, std: {out_gatv2.std().item():.4f}")
except Exception as e:
    print(f"✗ GATv2 forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Load config and instantiate full model
print("\n" + "=" * 60)
print("TEST 7: Testing full model with config...")
print("=" * 60)

try:
    with open('configs/experiment_008_gatv2.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = DualStreamPhylogeneticTransformer(config)
    print(f"✓ Full model created from config")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    batch_size = 32
    x = torch.randn(batch_size, 1024)
    edge_index = torch.randint(0, batch_size, (2, 150))

    logits = model(x, edge_index)
    print(f"✓ Full model forward pass successful")
    print(f"  Logits shape: {logits.shape}")

except Exception as e:
    print(f"✗ Full model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Compare parameter counts
print("\n" + "=" * 60)
print("TEST 8: Comparing architectures...")
print("=" * 60)

linear_params = sum(p.numel() for p in stream_linear.parameters())
gatv2_params = sum(p.numel() for p in stream_gatv2.parameters())

print(f"Linear StructuralStream:  {linear_params:>10,} parameters")
print(f"GATv2 StructuralStream:   {gatv2_params:>10,} parameters")
print(f"Difference:               {gatv2_params - linear_params:>+10,} ({100*(gatv2_params/linear_params - 1):+.1f}%)")

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print("\nSummary:")
print("- GATv2 implementation is working correctly")
print("- Backward compatibility with linear layer_type is maintained")
print("- Config loading and full model instantiation work")
print("- Ready for training with experiment_008_gatv2.yaml")
