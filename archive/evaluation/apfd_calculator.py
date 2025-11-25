"""
Modified APFD (Average Percentage of Faults Detected) Calculator
This module implements both classic and modified APFD calculations
Based on the project requirements and adapted from master_vini

Integration with filo_priori_v7:
- Compatible with existing code in src/evaluation/apfd.py
- Supports multiple APFD variants for comprehensive evaluation
- Handles different failure types (Fail, Blocked, Delete)
"""

import pandas as pd
import numpy as np
from typing import List, Union, Dict, Tuple
import ast


class APFDCalculator:
    """
    Enhanced APFD calculator with support for multiple failure types
    and weighted fault detection.
    """

    @staticmethod
    def calculate_classic_apfd(df_ordered: pd.DataFrame,
                              failure_types: List[str] = ['Fail']) -> float:
        """
        Calculate classic APFD metric.

        Args:
            df_ordered: DataFrame with test results, must contain 'TE_Test_Result' column
            failure_types: List of result types to consider as failures

        Returns:
            APFD score between 0 and 1
        """
        df_ordered = df_ordered.reset_index(drop=True)
        n = len(df_ordered)

        # Special case: single test case always returns 1.0
        # With only one test, there's no ordering to optimize
        if n == 1:
            return 1.0

        # Special case: no test cases
        if n == 0:
            return 1.0

        # Get failures based on specified types
        df_failures = df_ordered[df_ordered['TE_Test_Result'].isin(failure_types)]
        m = len(df_failures)

        # No failures means perfect APFD
        if m == 0:
            return 1.0

        # Get positions of failures (1-indexed)
        fail_indices = df_failures.index.tolist()
        tf_positions = [i + 1 for i in fail_indices]
        sum_positions = sum(tf_positions)

        # Classic APFD formula
        apfd = 1 - (sum_positions / (n * m)) + (1 / (2 * n))

        return apfd

    @staticmethod
    def calculate_weighted_apfd(df_ordered: pd.DataFrame,
                                failure_weights: Dict[str, float] = None) -> float:
        """
        Calculate weighted APFD where different failure types have different importance.

        Args:
            df_ordered: DataFrame with test results
            failure_weights: Dictionary mapping failure types to their weights
                           e.g., {'Fail': 1.0, 'Blocked': 0.5, 'Delete': 0.3}

        Returns:
            Weighted APFD score
        """
        if failure_weights is None:
            failure_weights = {'Fail': 1.0}

        df_ordered = df_ordered.reset_index(drop=True)
        n = len(df_ordered)

        # Special case: single test case always returns 1.0
        if n == 1:
            return 1.0

        if n == 0:
            return 1.0

        # Calculate weighted sum of fault positions
        weighted_sum = 0
        total_weight = 0

        for failure_type, weight in failure_weights.items():
            df_failures = df_ordered[df_ordered['TE_Test_Result'] == failure_type]
            m_type = len(df_failures)

            if m_type > 0:
                fail_indices = df_failures.index.tolist()
                tf_positions = [i + 1 for i in fail_indices]
                weighted_sum += weight * sum(tf_positions)
                total_weight += weight * m_type

        if total_weight == 0:
            return 1.0

        # Weighted APFD formula
        apfd = 1 - (weighted_sum / (n * total_weight)) + (1 / (2 * n))

        return apfd

    @staticmethod
    def calculate_napfd(df_ordered: pd.DataFrame,
                       failure_types: List[str] = ['Fail'],
                       test_costs: pd.Series = None) -> float:
        """
        Calculate Normalized APFD (NAPFD) considering test execution costs.
        This is useful when tests have different execution times.

        Args:
            df_ordered: DataFrame with test results
            failure_types: List of result types to consider as failures
            test_costs: Series with execution costs/times for each test

        Returns:
            NAPFD score
        """
        df_ordered = df_ordered.reset_index(drop=True)
        n = len(df_ordered)

        # Special case: single test case always returns 1.0
        if n == 1:
            return 1.0

        # If no costs provided, assume uniform cost of 1
        if test_costs is None:
            test_costs = pd.Series([1.0] * n)

        # Get failures
        df_failures = df_ordered[df_ordered['TE_Test_Result'].isin(failure_types)]
        m = len(df_failures)

        if m == 0 or n == 0:
            return 1.0

        # Calculate cumulative cost up to each failure
        cumulative_costs = []
        for idx in df_failures.index:
            cost_to_fault = test_costs.iloc[:idx+1].sum()
            cumulative_costs.append(cost_to_fault)

        total_cost = test_costs.sum()

        # NAPFD formula
        napfd = 1 - (sum(cumulative_costs) / (m * total_cost)) + (1 / (2 * m))

        return napfd

    @staticmethod
    def calculate_apfd_with_severity(df_ordered: pd.DataFrame,
                                    severity_column: str = 'severity',
                                    failure_types: List[str] = ['Fail']) -> float:
        """
        Calculate APFD considering fault severity levels.
        Higher severity faults detected earlier result in better APFD.

        Args:
            df_ordered: DataFrame with test results and severity information
            severity_column: Name of column containing severity scores
            failure_types: List of result types to consider as failures

        Returns:
            Severity-weighted APFD score
        """
        df_ordered = df_ordered.reset_index(drop=True)
        n = len(df_ordered)

        # Special case: single test case always returns 1.0
        if n == 1:
            return 1.0

        # Get failures
        df_failures = df_ordered[df_ordered['TE_Test_Result'].isin(failure_types)]

        if len(df_failures) == 0 or n == 0:
            return 1.0

        # If severity column doesn't exist, fall back to classic APFD
        if severity_column not in df_ordered.columns:
            return APFDCalculator.calculate_classic_apfd(df_ordered, failure_types)

        # Calculate severity-weighted positions
        weighted_positions = 0
        total_severity = 0

        for idx, row in df_failures.iterrows():
            position = idx + 1
            severity = row.get(severity_column, 1.0)
            weighted_positions += position * severity
            total_severity += severity

        if total_severity == 0:
            return 1.0

        # Severity-weighted APFD
        apfd = 1 - (weighted_positions / (n * total_severity)) + (1 / (2 * n))

        return apfd

    @staticmethod
    def calculate_apfd_c(df_ordered: pd.DataFrame,
                        failure_types: List[str] = ['Fail'],
                        confidence_threshold: float = 0.5) -> Tuple[float, float]:
        """
        Calculate APFD-C (APFD with Confidence) which also returns a confidence score.
        This is useful when the fault prediction has associated confidence values.

        Args:
            df_ordered: DataFrame with test results and 'prob_fail' column
            failure_types: List of result types to consider as failures
            confidence_threshold: Minimum confidence to consider a prediction reliable

        Returns:
            Tuple of (APFD score, confidence score)
        """
        df_ordered = df_ordered.reset_index(drop=True)

        # Calculate classic APFD
        apfd = APFDCalculator.calculate_classic_apfd(df_ordered, failure_types)

        # Calculate confidence if prob_fail column exists
        confidence = 1.0
        if 'prob_fail' in df_ordered.columns:
            # Get failures
            df_failures = df_ordered[df_ordered['TE_Test_Result'].isin(failure_types)]

            if len(df_failures) > 0:
                # Calculate average confidence for correctly predicted failures
                high_conf_failures = df_failures[df_failures['prob_fail'] >= confidence_threshold]
                if len(df_failures) > 0:
                    confidence = len(high_conf_failures) / len(df_failures)

        return apfd, confidence

    @staticmethod
    def calculate_modified_apfd(df_ordered: pd.DataFrame,
                               failure_types: List[str] = ['Fail', 'Blocked', 'Delete'],
                               use_weights: bool = True,
                               consider_severity: bool = False,
                               consider_costs: bool = False) -> Dict[str, float]:
        """
        Calculate the modified APFD as per project requirements.
        This is the main method that should be used in the project.

        Args:
            df_ordered: DataFrame with test results
            failure_types: Types of results to consider as failures
            use_weights: Whether to use weighted calculation for different failure types
            consider_severity: Whether to consider fault severity
            consider_costs: Whether to consider test execution costs

        Returns:
            Dictionary with different APFD metrics
        """
        results = {}

        # Classic APFD (only 'Fail' status)
        results['apfd_classic'] = APFDCalculator.calculate_classic_apfd(
            df_ordered, ['Fail']
        )

        # Extended APFD (all failure types)
        results['apfd_extended'] = APFDCalculator.calculate_classic_apfd(
            df_ordered, failure_types
        )

        # Weighted APFD if requested
        if use_weights:
            weights = {
                'Fail': 1.0,
                'Blocked': 0.7,  # Blocked tests are 70% as important as failures
                'Delete': 0.3    # Deleted tests are 30% as important
            }
            results['apfd_weighted'] = APFDCalculator.calculate_weighted_apfd(
                df_ordered, weights
            )

        # Severity-based APFD if requested and available
        if consider_severity:
            results['apfd_severity'] = APFDCalculator.calculate_apfd_with_severity(
                df_ordered, failure_types=failure_types
            )

        # Cost-based APFD if requested
        if consider_costs:
            # If time column exists, use it as cost
            if 'execution_time' in df_ordered.columns:
                costs = df_ordered['execution_time']
            else:
                costs = None
            results['napfd'] = APFDCalculator.calculate_napfd(
                df_ordered, failure_types, costs
            )

        # APFD with confidence
        if 'prob_fail' in df_ordered.columns:
            apfd_c, confidence = APFDCalculator.calculate_apfd_c(
                df_ordered, failure_types
            )
            results['apfd_confidence'] = apfd_c
            results['prediction_confidence'] = confidence

        # Primary metric for the project
        results['apfd'] = results.get('apfd_weighted', results['apfd_extended'])

        return results

    @staticmethod
    def count_total_commits(df_build: pd.DataFrame) -> int:
        """
        Count total commits for a build (including CRs).
        Works with both test.csv and test_filtered.csv structures.

        Args:
            df_build: DataFrame for a single build

        Returns:
            Total number of unique commits (including CRs)
        """
        total_commits = set()

        # Count commits from 'commit' column
        if 'commit' in df_build.columns:
            for commit_str in df_build['commit'].dropna():
                try:
                    commits = ast.literal_eval(commit_str)
                    if isinstance(commits, list):
                        total_commits.update(commits)
                    else:
                        total_commits.add(str(commit_str))
                except:
                    total_commits.add(str(commit_str))

        # Count CRs (works with both CR and CR_y columns)
        cr_column = 'CR_y' if 'CR_y' in df_build.columns else 'CR' if 'CR' in df_build.columns else None
        if cr_column:
            for cr_str in df_build[cr_column].dropna():
                try:
                    crs = ast.literal_eval(cr_str)
                    if isinstance(crs, list):
                        for cr in crs:
                            total_commits.add(f"CR_{cr}")
                except:
                    total_commits.add(f"CR_{cr_str}")

        return max(len(total_commits), 1)

    @staticmethod
    def filter_builds_with_failures(df: pd.DataFrame, only_with_failures: bool = True) -> pd.DataFrame:
        """
        Filter only builds that have at least one test with 'Fail' status.
        Only considers explicit 'Fail' results, excluding Blocked, Delete, and other statuses.

        Args:
            df: DataFrame with test results
            only_with_failures: If True, filter only builds with failures

        Returns:
            Filtered DataFrame
        """
        print("\n" + "="*60)
        print("Filtering builds with 'Fail' status")
        print("="*60)

        # Show distribution
        print("\nTest result distribution:")
        for result, count in df['TE_Test_Result'].value_counts().items():
            print(f"  {result}: {count:,} ({100*count/len(df):.1f}%)")

        # Identify builds with explicit failures only
        failure_types = ['Fail']  # Only consider 'Fail' status
        builds_with_issues = df[df['TE_Test_Result'].isin(failure_types)]['Build_ID'].unique()

        total_builds = df['Build_ID'].nunique()
        builds_with_failures = len(builds_with_issues)

        print(f"\nTotal builds: {total_builds}")
        print(f"Builds with 'Fail' status: {builds_with_failures}")
        print(f"Percentage with 'Fail': {100*builds_with_failures/total_builds:.1f}%")

        if only_with_failures:
            df_filtered = df[df['Build_ID'].isin(builds_with_issues)]
            print(f"Records after filter: {len(df_filtered)} of {len(df)} ({100*len(df_filtered)/len(df):.1f}%)")
            return df_filtered
        else:
            return df


def calculate_apfd(df_ordered: pd.DataFrame,
                  mode: str = 'modified',
                  **kwargs) -> Union[float, Dict[str, float]]:
    """
    Main function to calculate APFD based on project requirements.

    Args:
        df_ordered: DataFrame with ordered test results
        mode: 'classic' for traditional APFD, 'modified' for project-specific APFD
        **kwargs: Additional parameters for specific APFD calculations

    Returns:
        APFD score or dictionary of scores
    """
    calculator = APFDCalculator()

    if mode == 'classic':
        return calculator.calculate_classic_apfd(df_ordered, **kwargs)
    elif mode == 'modified':
        results = calculator.calculate_modified_apfd(df_ordered, **kwargs)
        # Return just the main APFD value for backward compatibility
        return results['apfd']
    elif mode == 'detailed':
        return calculator.calculate_modified_apfd(df_ordered, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'classic', 'modified', or 'detailed'")
