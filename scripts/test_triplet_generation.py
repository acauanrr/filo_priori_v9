"""
Test Triplet Generation

Quick script to test triplet generation on a small sample before full fine-tuning.

Usage:
    python scripts/test_triplet_generation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import logging

from embeddings.triplet_generator import create_triplet_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*70)
    logger.info("TESTING TRIPLET GENERATION")
    logger.info("="*70)

    # Load small sample
    logger.info("\nLoading 5000 sample rows...")
    df = pd.read_csv('datasets/train.csv', nrows=5000)

    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"Unique test cases: {df['TC_Key'].nunique()}")
    logger.info(f"Unique builds: {df['Build_ID'].nunique()}")

    # Check label distribution
    logger.info("\nLabel distribution:")
    print(df['TE_Test_Result'].value_counts())

    # Generate triplets
    logger.info("\n" + "="*70)
    logger.info("Generating triplets...")
    logger.info("="*70)

    triplets = create_triplet_dataset(
        df,
        min_fail_builds=1,
        min_pass_builds=1,
        max_triplets_per_test=5,
        output_path='cache/triplets_test.csv'
    )

    # Show examples
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE TRIPLETS")
    logger.info("="*70)

    for i, (anchor, positive, negative) in enumerate(triplets[:3]):
        print(f"\nTriplet {i+1}:")
        print(f"  Anchor (test case):")
        print(f"    {anchor[:150]}...")
        print(f"  Positive (commit from failed build):")
        print(f"    {positive[:150]}...")
        print(f"  Negative (commit from passed build):")
        print(f"    {negative[:150]}...")
        print()

    logger.info("="*70)
    logger.info("TEST COMPLETE")
    logger.info("="*70)
    logger.info(f"Generated {len(triplets)} triplets")
    logger.info("Saved to: cache/triplets_test.csv")
    logger.info("\nNext step: Run full fine-tuning with:")
    logger.info("  python scripts/finetune_bge.py --config configs/finetune_bge.yaml")

if __name__ == '__main__':
    main()
