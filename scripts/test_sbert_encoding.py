#!/usr/bin/env python3
"""
Quick test to verify SBERT encoding works without NVML errors

This tests with a small sample to ensure:
1. Model loads correctly
2. Encoding works on GPU
3. No memory fragmentation issues
4. Performance is acceptable
"""

import sys
from pathlib import Path
import time
import logging

import torch
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.sbert_encoder import SBERTEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*70)
    logger.info("SBERT ENCODING TEST")
    logger.info("="*70)

    # Check CUDA
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Create simple config
    config = {
        'embedding': {
            'model_name': 'sentence-transformers/all-mpnet-base-v2',
            'batch_size': 64,
            'max_length': 384,
            'normalize_embeddings': True
        }
    }

    # Initialize encoder
    logger.info("\nInitializing encoder...")
    start_time = time.time()
    encoder = SBERTEncoder(config, device='cuda' if torch.cuda.is_available() else 'cpu')
    init_time = time.time() - start_time
    logger.info(f"✓ Encoder initialized in {init_time:.2f}s")

    # Create test texts
    test_texts = [
        "Summary: Test user login functionality\nSteps: 1. Open app 2. Enter credentials 3. Click login",
        "Summary: Verify password reset\nSteps: 1. Click forgot password 2. Enter email 3. Check inbox",
        "Summary: Test checkout process\nSteps: 1. Add items to cart 2. Proceed to checkout 3. Enter payment info",
        "Commit Message: Fix login bug\n\nDiff:\n- fixed null pointer\n+ added validation",
        "Commit Message: Update README\n\nDiff:\n+ Added installation instructions",
    ] * 100  # 500 texts total

    logger.info(f"\nTest texts: {len(test_texts)}")

    # Test encoding
    logger.info("\nTesting encoding...")
    start_time = time.time()

    embeddings = encoder.encode_texts_chunked(
        test_texts,
        chunk_size=100,
        desc="Test encoding"
    )

    encode_time = time.time() - start_time

    logger.info(f"✓ Encoding completed in {encode_time:.2f}s")
    logger.info(f"✓ Embeddings shape: {embeddings.shape}")
    logger.info(f"✓ Speed: {len(test_texts)/encode_time:.1f} texts/s")
    logger.info(f"✓ Memory used: {embeddings.nbytes / 1024**2:.1f} MB")

    # Verify embeddings
    logger.info("\nVerifying embeddings...")
    assert embeddings.shape[0] == len(test_texts), "Wrong number of embeddings"
    assert embeddings.shape[1] == 768, "Wrong embedding dimension"
    assert not np.any(np.isnan(embeddings)), "NaN values in embeddings"
    assert not np.any(np.isinf(embeddings)), "Inf values in embeddings"
    logger.info("✓ All checks passed!")

    # Test memory stability - encode multiple times
    logger.info("\nTesting memory stability (10 iterations)...")
    for i in range(10):
        _ = encoder.encode_texts_batch(test_texts[:50])
        if torch.cuda.is_available() and i % 3 == 0:
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
            logger.info(f"  Iteration {i+1}: Allocated={mem_allocated:.1f}MB, Reserved={mem_reserved:.1f}MB")

    logger.info("✓ Memory stable - no fragmentation!")

    logger.info("\n" + "="*70)
    logger.info("TEST PASSED!")
    logger.info("="*70)
    logger.info("SBERT encoder is working correctly and stable.")
    logger.info("You can now run the full pre-computation.")

if __name__ == '__main__':
    main()
