"""
Test script for GAT integration (Step 2.4)

Tests:
1. StructuralStreamV8 initialization with GAT
2. Forward pass with edge_index and edge_weights
3. DualStreamModelV8 with graph structure
"""

import torch
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_structural_stream_gat():
    """Test 1: StructuralStreamV8 with GAT"""
    logger.info("="*70)
    logger.info("TEST 1: StructuralStreamV8 with GAT")
    logger.info("="*70)

    try:
        from src.models.dual_stream_v8 import StructuralStreamV8

        # Create simple graph
        # 5 nodes, edges: 0->1, 1->2, 2->3, 3->4, 4->0 (cycle)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4],  # source nodes
            [1, 2, 3, 4, 0]   # target nodes
        ], dtype=torch.long)

        edge_weights = torch.tensor([0.8, 0.6, 0.7, 0.9, 0.5], dtype=torch.float32)

        # Create dummy features [5 nodes, 6 features]
        x = torch.randn(5, 6)

        # Initialize GAT-based structural stream
        logger.info("\nInitializing GAT-based StructuralStreamV8...")
        stream = StructuralStreamV8(
            input_dim=6,
            hidden_dim=256,
            num_heads=4,
            dropout=0.3,
            activation='elu',
            use_edge_weights=True
        )

        # Forward pass
        logger.info("Running forward pass...")
        stream.eval()
        with torch.no_grad():
            output = stream(x, edge_index, edge_weights)

        logger.info(f"✓ Input shape: {x.shape}")
        logger.info(f"✓ Edge_index shape: {edge_index.shape}")
        logger.info(f"✓ Edge_weights shape: {edge_weights.shape}")
        logger.info(f"✓ Output shape: {output.shape}")
        logger.info(f"✓ Expected output: [5, 256]")

        assert output.shape == (5, 256), f"Output shape mismatch: {output.shape} != (5, 256)"

        logger.info("\n✅ TEST 1 PASSED: StructuralStreamV8 with GAT works!\n")
        return True

    except Exception as e:
        logger.error(f"\n❌ TEST 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_dual_stream_with_graph():
    """Test 2: DualStreamModelV8 with graph structure"""
    logger.info("="*70)
    logger.info("TEST 2: DualStreamModelV8 with graph structure")
    logger.info("="*70)

    try:
        from src.models.dual_stream_v8 import DualStreamModelV8

        # Create simple graph
        batch_size = 8
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 0]
        ], dtype=torch.long)

        edge_weights = torch.rand(8)

        # Create dummy inputs
        semantic_input = torch.randn(batch_size, 1024)  # BGE embeddings
        structural_input = torch.randn(batch_size, 6)    # Historical features

        # Initialize model
        logger.info("\nInitializing DualStreamModelV8...")
        config = {
            'semantic': {
                'input_dim': 1024,
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.3,
                'activation': 'gelu'
            },
            'structural': {
                'input_dim': 6,
                'hidden_dim': 256,
                'num_heads': 4,
                'dropout': 0.3,
                'activation': 'elu',
                'use_edge_weights': True
            },
            'fusion': {
                'num_heads': 4,
                'dropout': 0.1
            },
            'classifier': {
                'hidden_dims': [128, 64],
                'dropout': 0.4
            },
            'num_classes': 2
        }

        model = DualStreamModelV8(
            semantic_config=config['semantic'],
            structural_config=config['structural'],
            fusion_config=config['fusion'],
            classifier_config=config['classifier'],
            num_classes=config['num_classes']
        )

        # Forward pass
        logger.info("\nRunning forward pass...")
        model.eval()
        with torch.no_grad():
            logits = model(
                semantic_input=semantic_input,
                structural_input=structural_input,
                edge_index=edge_index,
                edge_weights=edge_weights
            )

        logger.info(f"✓ Semantic input shape: {semantic_input.shape}")
        logger.info(f"✓ Structural input shape: {structural_input.shape}")
        logger.info(f"✓ Edge_index shape: {edge_index.shape}")
        logger.info(f"✓ Edge_weights shape: {edge_weights.shape}")
        logger.info(f"✓ Output logits shape: {logits.shape}")
        logger.info(f"✓ Expected output: [{batch_size}, 2]")

        assert logits.shape == (batch_size, 2), f"Output shape mismatch: {logits.shape} != ({batch_size}, 2)"

        logger.info("\n✅ TEST 2 PASSED: DualStreamModelV8 with graph works!\n")
        return True

    except Exception as e:
        logger.error(f"\n❌ TEST 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_model_parameter_count():
    """Test 3: Check parameter count and model architecture"""
    logger.info("="*70)
    logger.info("TEST 3: Model architecture and parameters")
    logger.info("="*70)

    try:
        from src.models.dual_stream_v8 import DualStreamModelV8

        config = {
            'semantic': {'input_dim': 1024, 'hidden_dim': 256},
            'structural': {'input_dim': 6, 'hidden_dim': 256, 'num_heads': 4},
            'fusion': {'num_heads': 4},
            'classifier': {'hidden_dims': [128, 64]},
            'num_classes': 2
        }

        model = DualStreamModelV8(
            semantic_config=config['semantic'],
            structural_config=config['structural'],
            fusion_config=config['fusion'],
            classifier_config=config['classifier'],
            num_classes=config['num_classes']
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"\n✓ Total parameters: {total_params:,}")
        logger.info(f"✓ Trainable parameters: {trainable_params:,}")
        logger.info(f"✓ Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

        # Check that model has GAT components
        has_gat = any('conv1' in name or 'conv2' in name for name, _ in model.named_modules())
        logger.info(f"✓ Has GAT layers: {has_gat}")

        assert has_gat, "Model should have GAT layers (conv1, conv2)"
        assert total_params > 0, "Model should have parameters"

        logger.info("\n✅ TEST 3 PASSED: Model architecture is correct!\n")
        return True

    except Exception as e:
        logger.error(f"\n❌ TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "="*70)
    logger.info("GAT INTEGRATION TESTS (Step 2.4)")
    logger.info("="*70 + "\n")

    results = []

    # Test 1: StructuralStreamV8 with GAT
    results.append(("StructuralStreamV8 with GAT", test_structural_stream_gat()))

    # Test 2: DualStreamModelV8 with graph
    results.append(("DualStreamModelV8 with graph", test_dual_stream_with_graph()))

    # Test 3: Model architecture
    results.append(("Model architecture", test_model_parameter_count()))

    # Summary
    logger.info("="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    logger.info(f"\nTotal: {passed}/{total} tests passed")
    logger.info("="*70 + "\n")

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! GAT integration is working correctly.\n")
        return 0
    else:
        logger.error("⚠️  SOME TESTS FAILED. Please check the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
