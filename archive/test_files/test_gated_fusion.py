"""
Test script for Gated Fusion Unit (Step 2.5)

Tests:
1. GatedFusionUnit forward pass
2. Gate behavior with zero structural features
3. DualStreamModelV8 with gated fusion
4. Comparison: cross-attention vs gated fusion
"""

import torch
import sys
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_gated_fusion_basic():
    """Test 1: Basic GatedFusionUnit functionality"""
    logger.info("="*70)
    logger.info("TEST 1: GatedFusionUnit Basic Functionality")
    logger.info("="*70)

    try:
        from src.models.dual_stream_v8 import GatedFusionUnit

        batch_size = 8
        hidden_dim = 256

        # Create dummy features
        semantic_features = torch.randn(batch_size, hidden_dim)
        structural_features = torch.randn(batch_size, hidden_dim)

        # Initialize GFU
        logger.info("\nInitializing GatedFusionUnit...")
        gfu = GatedFusionUnit(
            hidden_dim=hidden_dim,
            dropout=0.1,
            use_projection=True
        )

        # Forward pass
        logger.info("Running forward pass...")
        gfu.eval()
        with torch.no_grad():
            output = gfu(semantic_features, structural_features)

        logger.info(f"✓ Semantic input shape: {semantic_features.shape}")
        logger.info(f"✓ Structural input shape: {structural_features.shape}")
        logger.info(f"✓ Output shape: {output.shape}")
        logger.info(f"✓ Expected output: [{batch_size}, {hidden_dim * 2}]")

        assert output.shape == (batch_size, hidden_dim * 2), \
            f"Output shape mismatch: {output.shape} != ({batch_size}, {hidden_dim * 2})"

        logger.info("\n✅ TEST 1 PASSED: GatedFusionUnit works correctly!\n")
        return True

    except Exception as e:
        logger.error(f"\n❌ TEST 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_gate_with_zero_structural():
    """Test 2: Gate behavior with zero structural features (sparse data)"""
    logger.info("="*70)
    logger.info("TEST 2: Gate Behavior with Zero Structural Features")
    logger.info("="*70)

    try:
        from src.models.dual_stream_v8 import GatedFusionUnit

        batch_size = 4
        hidden_dim = 256

        # Create features: semantic has signal, structural is zero (new test case)
        semantic_features = torch.randn(batch_size, hidden_dim)
        structural_features = torch.zeros(batch_size, hidden_dim)  # No history!

        # Initialize GFU
        logger.info("\nScenario: New test case with NO execution history")
        logger.info("  Semantic features: random (has signal)")
        logger.info("  Structural features: ZERO (no history)")
        logger.info("\nExpected behavior:")
        logger.info("  Gate should learn to suppress structural (rely on semantic)")

        gfu = GatedFusionUnit(hidden_dim=hidden_dim, dropout=0.0, use_projection=False)

        # Forward pass
        gfu.eval()
        with torch.no_grad():
            output_zero_struct = gfu(semantic_features, structural_features)

        # Compare with both having signal
        structural_features_signal = torch.randn(batch_size, hidden_dim)
        with torch.no_grad():
            output_both_signal = gfu(semantic_features, structural_features_signal)

        logger.info(f"\n✓ Output with zero structural: shape {output_zero_struct.shape}")
        logger.info(f"✓ Output with both signals: shape {output_both_signal.shape}")

        # Check that outputs are different (gate is working)
        diff = torch.abs(output_zero_struct - output_both_signal).mean().item()
        logger.info(f"✓ Mean difference between outputs: {diff:.4f}")

        assert diff > 0.01, "Gate should produce different outputs for zero vs signal"

        logger.info("\n✅ TEST 2 PASSED: Gate responds to structural sparsity!\n")
        return True

    except Exception as e:
        logger.error(f"\n❌ TEST 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_model_with_gated_fusion():
    """Test 3: DualStreamModelV8 with gated fusion"""
    logger.info("="*70)
    logger.info("TEST 3: DualStreamModelV8 with Gated Fusion")
    logger.info("="*70)

    try:
        from src.models.dual_stream_v8 import DualStreamModelV8

        batch_size = 8
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                   [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
        edge_weights = torch.rand(8)

        semantic_input = torch.randn(batch_size, 1024)
        structural_input = torch.randn(batch_size, 6)

        # Initialize model with gated fusion
        logger.info("\nInitializing DualStreamModelV8 with GATED fusion...")
        config = {
            'semantic': {'input_dim': 1024, 'hidden_dim': 256},
            'structural': {'input_dim': 6, 'hidden_dim': 256, 'num_heads': 4},
            'fusion': {
                'type': 'gated',  # Use gated fusion!
                'dropout': 0.1,
                'use_projection': True
            },
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

        # Forward pass
        logger.info("\nRunning forward pass...")
        model.eval()
        with torch.no_grad():
            logits = model(semantic_input, structural_input, edge_index, edge_weights)

        logger.info(f"✓ Output logits shape: {logits.shape}")
        logger.info(f"✓ Expected: [{batch_size}, 2]")

        assert logits.shape == (batch_size, 2), f"Shape mismatch: {logits.shape}"

        # Check that fusion is GatedFusionUnit
        from src.models.dual_stream_v8 import GatedFusionUnit
        assert isinstance(model.fusion, GatedFusionUnit), \
            f"Fusion should be GatedFusionUnit, got {type(model.fusion)}"

        logger.info(f"✓ Fusion layer type: {type(model.fusion).__name__}")

        logger.info("\n✅ TEST 3 PASSED: Model with gated fusion works!\n")
        return True

    except Exception as e:
        logger.error(f"\n❌ TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_fusion_comparison():
    """Test 4: Compare cross-attention vs gated fusion"""
    logger.info("="*70)
    logger.info("TEST 4: Cross-Attention vs Gated Fusion Comparison")
    logger.info("="*70)

    try:
        from src.models.dual_stream_v8 import DualStreamModelV8

        batch_size = 8
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                   [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
        edge_weights = torch.rand(8)

        semantic_input = torch.randn(batch_size, 1024)
        structural_input = torch.randn(batch_size, 6)

        # Model 1: Cross-Attention Fusion
        logger.info("\n[1] Creating model with CROSS-ATTENTION fusion...")
        config_cross = {
            'semantic': {'input_dim': 1024, 'hidden_dim': 256},
            'structural': {'input_dim': 6, 'hidden_dim': 256, 'num_heads': 4},
            'fusion': {'type': 'cross_attention', 'num_heads': 4, 'dropout': 0.1},
            'classifier': {'hidden_dims': [128, 64]},
            'num_classes': 2
        }

        model_cross = DualStreamModelV8(**{k+'_config' if k != 'num_classes' else k: v
                                          for k, v in config_cross.items()})

        # Model 2: Gated Fusion
        logger.info("[2] Creating model with GATED fusion...")
        config_gated = config_cross.copy()
        config_gated['fusion'] = {'type': 'gated', 'dropout': 0.1, 'use_projection': True}

        model_gated = DualStreamModelV8(**{k+'_config' if k != 'num_classes' else k: v
                                          for k, v in config_gated.items()})

        # Compare parameter counts
        params_cross = sum(p.numel() for p in model_cross.parameters())
        params_gated = sum(p.numel() for p in model_gated.parameters())

        logger.info(f"\n✓ Parameters (Cross-Attention): {params_cross:,}")
        logger.info(f"✓ Parameters (Gated Fusion): {params_gated:,}")
        logger.info(f"✓ Difference: {abs(params_cross - params_gated):,}")

        # Forward pass both models
        model_cross.eval()
        model_gated.eval()

        with torch.no_grad():
            logits_cross = model_cross(semantic_input, structural_input, edge_index, edge_weights)
            logits_gated = model_gated(semantic_input, structural_input, edge_index, edge_weights)

        logger.info(f"\n✓ Cross-Attention output: {logits_cross.shape}")
        logger.info(f"✓ Gated Fusion output: {logits_gated.shape}")

        # Both should produce valid outputs
        assert logits_cross.shape == logits_gated.shape == (batch_size, 2)

        logger.info("\n✅ TEST 4 PASSED: Both fusion types work correctly!\n")
        return True

    except Exception as e:
        logger.error(f"\n❌ TEST 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "="*70)
    logger.info("GATED FUSION UNIT TESTS (Step 2.5)")
    logger.info("="*70 + "\n")

    results = []

    # Test 1: Basic GFU functionality
    results.append(("GatedFusionUnit basic", test_gated_fusion_basic()))

    # Test 2: Gate behavior with zero structural
    results.append(("Gate with zero structural", test_gate_with_zero_structural()))

    # Test 3: Model with gated fusion
    results.append(("DualStreamModelV8 with GFU", test_model_with_gated_fusion()))

    # Test 4: Comparison
    results.append(("Cross-Attention vs Gated", test_fusion_comparison()))

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
        logger.info("🎉 ALL TESTS PASSED! Gated Fusion integration is working correctly.\n")
        return 0
    else:
        logger.error("⚠️  SOME TESTS FAILED. Please check the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
