#!/usr/bin/env python3
"""
SBERT Embedding Pre-computation Script

Uses all-mpnet-base-v2 instead of Qodo-Embed-1-1.5B
- 13x smaller model (110M vs 1.5B parameters)
- 15x less VRAM (~200MB vs 3GB)
- No NVML/memory fragmentation issues
- Excellent performance for text embeddings

Usage:
    python scripts/precompute_embeddings_sbert.py \
        --config configs/experiment_sbert.yaml \
        --output cache/embeddings_sbert.npz \
        --batch_size 64
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import time

import numpy as np
import yaml
import torch
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.sbert_encoder import SBERTEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sbert_encoding.log')
    ]
)
logger = logging.getLogger(__name__)


def load_data(config_path: str):
    """Load train and test datasets"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    train_path = data_config.get('train_path') or data_config.get('train_dataset_path')
    test_path = data_config.get('test_path') or data_config.get('test_dataset_path')

    logger.info(f"Loading train data: {train_path}")
    train_df = pd.read_csv(train_path)

    logger.info(f"Loading test data: {test_path}")
    test_df = pd.read_csv(test_path)

    logger.info(f"✓ Train: {len(train_df)} samples")
    logger.info(f"✓ Test: {len(test_df)} samples")

    return train_df, test_df, config


def prepare_tc_texts(df: pd.DataFrame) -> list:
    """Prepare test case texts from dataframe"""
    texts = []
    for _, row in df.iterrows():
        summary = row.get('tc_summary', row.get('summary', ''))
        steps = row.get('tc_steps', row.get('steps', ''))

        # Combine summary and steps
        if summary and steps:
            text = f"Summary: {summary}\nSteps: {steps}"
        elif summary:
            text = f"Summary: {summary}"
        elif steps:
            text = f"Steps: {steps}"
        else:
            text = "No test case information"

        texts.append(text)

    return texts


def prepare_commit_texts(df: pd.DataFrame) -> list:
    """Prepare commit texts from dataframe"""
    texts = []
    for _, row in df.iterrows():
        msg = row.get('commit_msg', row.get('message', ''))
        diff = row.get('commit_diff', row.get('diff', ''))

        # Combine message and diff (truncate diff if too long)
        if msg and diff:
            # Truncate diff to 2000 chars (SBERT max is 512 tokens, ~2000 chars)
            diff_truncated = diff[:2000] if len(diff) > 2000 else diff
            text = f"Commit Message: {msg}\n\nDiff:\n{diff_truncated}"
        elif msg:
            text = f"Commit Message: {msg}"
        elif diff:
            diff_truncated = diff[:2000] if len(diff) > 2000 else diff
            text = f"Diff:\n{diff_truncated}"
        else:
            text = "No commit information"

        texts.append(text)

    return texts


def main():
    parser = argparse.ArgumentParser(description='SBERT embedding pre-computation')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--output', type=str, required=True, help='Output .npz file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--chunk_size', type=int, default=640, help='Chunk size (default: 640)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("SBERT EMBEDDING PRE-COMPUTATION")
    logger.info("="*70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Device: {args.device}")

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load data
    logger.info("\n" + "="*70)
    logger.info("LOADING DATA")
    logger.info("="*70)
    train_df, test_df, config = load_data(args.config)

    # Update config with args
    if 'embedding' not in config:
        config['embedding'] = {}
    config['embedding']['batch_size'] = args.batch_size
    config['embedding']['model_name'] = 'sentence-transformers/all-mpnet-base-v2'

    # Initialize encoder
    logger.info("\n" + "="*70)
    logger.info("INITIALIZING ENCODER")
    logger.info("="*70)
    encoder = SBERTEncoder(config, device=args.device)

    # Prepare texts
    logger.info("\n" + "="*70)
    logger.info("PREPARING TEXTS")
    logger.info("="*70)

    train_tc_texts = prepare_tc_texts(train_df)
    test_tc_texts = prepare_tc_texts(test_df)
    train_commit_texts = prepare_commit_texts(train_df)
    test_commit_texts = prepare_commit_texts(test_df)

    logger.info(f"Train TCs: {len(train_tc_texts)}")
    logger.info(f"Test TCs: {len(test_tc_texts)}")
    logger.info(f"Train Commits: {len(train_commit_texts)}")
    logger.info(f"Test Commits: {len(test_commit_texts)}")

    # Encode
    logger.info("\n" + "="*70)
    logger.info("ENCODING")
    logger.info("="*70)

    total_start = time.time()

    train_tc_emb = encoder.encode_texts_chunked(
        train_tc_texts,
        chunk_size=args.chunk_size,
        desc="Train TCs"
    )

    test_tc_emb = encoder.encode_texts_chunked(
        test_tc_texts,
        chunk_size=args.chunk_size,
        desc="Test TCs"
    )

    train_commit_emb = encoder.encode_texts_chunked(
        train_commit_texts,
        chunk_size=args.chunk_size,
        desc="Train Commits"
    )

    test_commit_emb = encoder.encode_texts_chunked(
        test_commit_texts,
        chunk_size=args.chunk_size,
        desc="Test Commits"
    )

    total_elapsed = time.time() - total_start

    # Save
    logger.info("\n" + "="*70)
    logger.info("SAVING")
    logger.info("="*70)

    np.savez_compressed(
        args.output,
        train_tc_embeddings=train_tc_emb,
        test_tc_embeddings=test_tc_emb,
        train_commit_embeddings=train_commit_emb,
        test_commit_embeddings=test_commit_emb,
        embedding_dim=encoder.embedding_dim,
        model_name=encoder.model_name
    )

    file_size_mb = os.path.getsize(args.output) / (1024 ** 2)
    total_samples = len(train_tc_texts) + len(test_tc_texts) + len(train_commit_texts) + len(test_commit_texts)

    logger.info(f"✓ Saved to: {args.output}")
    logger.info(f"✓ File size: {file_size_mb:.1f} MB")
    logger.info(f"✓ Total time: {total_elapsed/60:.1f} minutes")
    logger.info(f"✓ Total samples: {total_samples}")
    logger.info(f"✓ Average speed: {total_samples/total_elapsed:.1f} samples/s")
    logger.info(f"✓ Embedding dimension: {encoder.embedding_dim}")

    logger.info("\n" + "="*70)
    logger.info("SUCCESS!")
    logger.info("="*70)


if __name__ == '__main__':
    main()
