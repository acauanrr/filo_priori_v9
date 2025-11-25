"""
Triplet Generator for Contrastive Fine-Tuning of BGE Embeddings

This module generates triplets (anchor, positive, negative) for contrastive learning,
training the embedder to understand domain-specific relationships between test cases
and code changes in software engineering.

Triplet Strategy:
- Anchor: Test case text (TE_Summary + TC_Steps)
- Positive: Commit text from builds where this test FAILED
- Negative: Commit text from builds where this test PASSED

Objective: Force test text and failure-causing commit text to be close in embedding space.

Author: Filo-Priori V8 Team
Date: 2025-11-06
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import logging
import ast

logger = logging.getLogger(__name__)


class TripletGenerator:
    """
    Generates triplets for contrastive learning from test execution history.

    Each triplet consists of:
    - Anchor: Test case description (Summary + Steps)
    - Positive: Commit message from a build where the test failed
    - Negative: Commit message from a build where the test passed
    """

    def __init__(self,
                 min_fail_builds: int = 1,
                 min_pass_builds: int = 1,
                 max_triplets_per_test: int = 10,
                 seed: int = 42):
        """
        Args:
            min_fail_builds: Minimum number of failed builds required for a test
            min_pass_builds: Minimum number of passed builds required for a test
            max_triplets_per_test: Maximum triplets to generate per test case
            seed: Random seed for reproducibility
        """
        self.min_fail_builds = min_fail_builds
        self.min_pass_builds = min_pass_builds
        self.max_triplets_per_test = max_triplets_per_test
        self.seed = seed
        np.random.seed(seed)

        # Storage for test case history
        self.test_history = defaultdict(lambda: {'failed': [], 'passed': []})

    def _parse_commit_field(self, commit_str: str) -> str:
        """
        Parse commit field which may be in list/array format.

        Args:
            commit_str: Commit field (may be string representation of list)

        Returns:
            Cleaned commit text
        """
        if pd.isna(commit_str):
            return ""

        try:
            # Try to parse as list
            commit_list = ast.literal_eval(str(commit_str))
            if isinstance(commit_list, list):
                # Join first 3 commits
                commits = [str(c).strip() for c in commit_list[:3] if c]
                return " [SEP] ".join(commits)
        except:
            pass

        # Return as string if parsing fails
        return str(commit_str).strip()

    def _create_anchor_text(self, summary: str, steps: str) -> str:
        """
        Create anchor text from test case summary and steps.

        Args:
            summary: Test execution summary
            steps: Test case steps

        Returns:
            Combined anchor text
        """
        summary = str(summary).strip() if not pd.isna(summary) else ""
        steps = str(steps).strip() if not pd.isna(steps) else ""

        if summary and steps:
            return f"{summary} [SEP] {steps}"
        elif summary:
            return summary
        elif steps:
            return steps
        else:
            return "[EMPTY]"

    def fit(self, df: pd.DataFrame):
        """
        Build test case history from training data.

        Args:
            df: Training DataFrame with columns:
                - TC_Key: Test case identifier
                - TE_Summary: Test execution summary
                - TC_Steps: Test case steps
                - commit: Commit messages
                - TE_Test_Result: Test result (Pass/Fail/etc.)
        """
        logger.info("="*70)
        logger.info("BUILDING TRIPLET DATASET FROM TEST HISTORY")
        logger.info("="*70)

        logger.info(f"Processing {len(df)} test executions...")

        # Group by test case
        for tc_key, group in df.groupby('TC_Key'):
            # Get anchor text (use first occurrence for consistency)
            first_row = group.iloc[0]
            anchor_text = self._create_anchor_text(
                first_row['TE_Summary'],
                first_row['TC_Steps']
            )

            # Collect commits from failed and passed builds
            for _, row in group.iterrows():
                commit_text = self._parse_commit_field(row['commit'])
                if not commit_text or commit_text == "[EMPTY]":
                    continue

                result = row['TE_Test_Result']

                if result == 'Fail':
                    self.test_history[tc_key]['failed'].append({
                        'anchor': anchor_text,
                        'commit': commit_text,
                        'build_id': row['Build_ID']
                    })
                elif result == 'Pass':
                    self.test_history[tc_key]['passed'].append({
                        'anchor': anchor_text,
                        'commit': commit_text,
                        'build_id': row['Build_ID']
                    })

        # Filter tests with insufficient history
        valid_tests = []
        for tc_key, history in self.test_history.items():
            n_fail = len(history['failed'])
            n_pass = len(history['passed'])

            if n_fail >= self.min_fail_builds and n_pass >= self.min_pass_builds:
                valid_tests.append(tc_key)

        logger.info(f"\nTest Case History:")
        logger.info(f"  Total test cases: {len(self.test_history)}")
        logger.info(f"  Valid for triplets: {len(valid_tests)}")
        logger.info(f"  Min fail builds: {self.min_fail_builds}")
        logger.info(f"  Min pass builds: {self.min_pass_builds}")

        # Keep only valid tests
        self.test_history = {k: v for k, v in self.test_history.items() if k in valid_tests}

        return self

    def generate_triplets(self) -> List[Tuple[str, str, str]]:
        """
        Generate triplets for contrastive learning.

        Returns:
            List of (anchor, positive, negative) tuples where:
            - anchor: Test case text
            - positive: Commit from a failed build
            - negative: Commit from a passed build
        """
        triplets = []

        logger.info("\nGenerating triplets...")

        for tc_key, history in self.test_history.items():
            failed = history['failed']
            passed = history['passed']

            # Generate up to max_triplets_per_test triplets
            n_triplets = min(
                self.max_triplets_per_test,
                len(failed) * len(passed)  # All possible combinations
            )

            for _ in range(n_triplets):
                # Sample positive (failed build)
                pos_sample = np.random.choice(failed)
                anchor = pos_sample['anchor']
                positive = pos_sample['commit']

                # Sample negative (passed build)
                neg_sample = np.random.choice(passed)
                negative = neg_sample['commit']

                triplets.append((anchor, positive, negative))

        logger.info(f"Generated {len(triplets)} triplets from {len(self.test_history)} test cases")
        logger.info(f"Avg triplets per test: {len(triplets) / len(self.test_history):.1f}")

        return triplets

    def get_statistics(self) -> Dict:
        """
        Get statistics about the triplet dataset.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_tests': len(self.test_history),
            'total_failed_instances': sum(len(h['failed']) for h in self.test_history.values()),
            'total_passed_instances': sum(len(h['passed']) for h in self.test_history.values()),
            'avg_failed_per_test': 0.0,
            'avg_passed_per_test': 0.0,
        }

        if stats['num_tests'] > 0:
            stats['avg_failed_per_test'] = stats['total_failed_instances'] / stats['num_tests']
            stats['avg_passed_per_test'] = stats['total_passed_instances'] / stats['num_tests']

        return stats


def create_triplet_dataset(
    df_train: pd.DataFrame,
    min_fail_builds: int = 1,
    min_pass_builds: int = 1,
    max_triplets_per_test: int = 10,
    output_path: Optional[str] = None
) -> List[Tuple[str, str, str]]:
    """
    Convenience function to create triplet dataset from training data.

    Args:
        df_train: Training DataFrame
        min_fail_builds: Minimum failed builds per test
        min_pass_builds: Minimum passed builds per test
        max_triplets_per_test: Maximum triplets per test
        output_path: Optional path to save triplets as CSV

    Returns:
        List of (anchor, positive, negative) triplets
    """
    generator = TripletGenerator(
        min_fail_builds=min_fail_builds,
        min_pass_builds=min_pass_builds,
        max_triplets_per_test=max_triplets_per_test
    )

    # Build history
    generator.fit(df_train)

    # Generate triplets
    triplets = generator.generate_triplets()

    # Print statistics
    stats = generator.get_statistics()
    logger.info("\n" + "="*70)
    logger.info("TRIPLET DATASET STATISTICS")
    logger.info("="*70)
    logger.info(f"Test cases with valid history: {stats['num_tests']}")
    logger.info(f"Total failed instances: {stats['total_failed_instances']}")
    logger.info(f"Total passed instances: {stats['total_passed_instances']}")
    logger.info(f"Avg failed builds per test: {stats['avg_failed_per_test']:.2f}")
    logger.info(f"Avg passed builds per test: {stats['avg_passed_per_test']:.2f}")
    logger.info(f"Total triplets generated: {len(triplets)}")
    logger.info("="*70)

    # Save to file if requested
    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df_triplets = pd.DataFrame(triplets, columns=['anchor', 'positive', 'negative'])
        df_triplets.to_csv(output_path, index=False)
        logger.info(f"\nTriplets saved to: {output_path}")

    return triplets


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Load data
    df = pd.read_csv('datasets/train.csv').head(5000)

    # Generate triplets
    triplets = create_triplet_dataset(
        df,
        min_fail_builds=1,
        min_pass_builds=1,
        max_triplets_per_test=5,
        output_path='data/triplets_sample.csv'
    )

    # Show examples
    print("\nExample triplets:")
    for i, (anchor, pos, neg) in enumerate(triplets[:3]):
        print(f"\nTriplet {i+1}:")
        print(f"  Anchor: {anchor[:100]}...")
        print(f"  Positive: {pos[:100]}...")
        print(f"  Negative: {neg[:100]}...")
