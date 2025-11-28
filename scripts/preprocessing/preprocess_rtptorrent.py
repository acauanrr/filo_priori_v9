#!/usr/bin/env python3
"""
Preprocess RTPTorrent Dataset for Filo-Priori.

This script converts the RTPTorrent dataset to the format expected by Filo-Priori.

Usage:
    python scripts/preprocessing/preprocess_rtptorrent.py [--project PROJECT_NAME]

Options:
    --project: Process only a specific project (default: all projects)
    --sample: Create a smaller sample for testing
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
RAW_DIR = BASE_DIR / "datasets" / "02_rtptorrent" / "raw"
OUTPUT_DIR = BASE_DIR / "datasets" / "02_rtptorrent" / "processed"


def find_rtptorrent_data(raw_dir: Path) -> Optional[Path]:
    """Find the RTPTorrent data directory."""
    # Look for extracted folder
    candidates = [
        raw_dir / "MSR2",  # Actual folder name in the zip
        raw_dir / "rtp-torrent-v1",
        raw_dir / "rtp-torrent",
        raw_dir / "RTPTorrent",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Check if data is directly in raw_dir
    if (raw_dir / "builds").exists() or any(raw_dir.glob("*.csv")):
        return raw_dir

    return None


def list_available_projects(data_dir: Path) -> List[str]:
    """List available projects in the RTPTorrent dataset."""
    projects = []

    # Look for project directories (format: owner@repo)
    for item in data_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # RTPTorrent projects have format owner@repo
            if '@' in item.name:
                projects.append(item.name)
            # Also check for CSV files as fallback
            elif any(item.glob("*.csv")) or (item / "builds").exists():
                projects.append(item.name)

    return sorted(projects)


def parse_test_name(test_name: str) -> Tuple[str, str, str]:
    """
    Parse a fully qualified test name into package, class, and method.

    Example: "org.apache.commons.math3.analysis.function.SincTest::testValue"
    Returns: ("org.apache.commons.math3.analysis.function", "SincTest", "testValue")
    """
    # Split by :: to get class and method
    if "::" in test_name:
        class_part, method = test_name.rsplit("::", 1)
    else:
        class_part = test_name
        method = ""

    # Split class part into package and class name
    if "." in class_part:
        package, class_name = class_part.rsplit(".", 1)
    else:
        package = ""
        class_name = class_part

    return package, class_name, method


def generate_test_description(package: str, class_name: str, method: str) -> str:
    """Generate a human-readable test description from test name components."""
    # Convert camelCase to words
    def camel_to_words(name: str) -> str:
        # Insert space before capitals
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        # Handle consecutive capitals
        words = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', words)
        return words.lower()

    # Clean method name
    method_words = camel_to_words(method.replace("test", "").replace("Test", ""))

    # Clean class name
    class_words = camel_to_words(class_name.replace("Test", ""))

    # Build description
    if method_words.strip():
        description = f"Test {method_words.strip()} in {class_words.strip()}"
    else:
        description = f"Test {class_words.strip()}"

    return description


def load_project_data(project_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load test execution data for a project.

    Returns: (test_executions_df, builds_df)
    """
    # Find CSV files - RTPTorrent uses format: project@repo.csv
    full_csv = None
    builds_csv = None
    project_name = project_dir.name

    # Look for main data file (format: owner@repo.csv)
    for csv_file in project_dir.glob("*.csv"):
        name = csv_file.name.lower()
        # Skip auxiliary files
        if any(x in name for x in ['offenders', 'patches', 'pr', 'commits', 'job']):
            continue
        # Main data file matches project name pattern
        if '@' in csv_file.name and not any(x in name for x in ['-offenders', '-patches', '-pr']):
            full_csv = csv_file
            break

    if full_csv is None:
        # Try alternative structures
        for csv_file in project_dir.glob("*.csv"):
            if "full" in csv_file.name.lower():
                full_csv = csv_file
                break
        if full_csv is None and (project_dir / "test_executions.csv").exists():
            full_csv = project_dir / "test_executions.csv"
        elif full_csv is None and (project_dir / "data.csv").exists():
            full_csv = project_dir / "data.csv"

    if full_csv is None:
        raise FileNotFoundError(f"No test execution CSV found in {project_dir}")

    # Load data
    print(f"  Loading: {full_csv.name}")
    test_df = pd.read_csv(full_csv)

    builds_df = None
    if builds_csv:
        print(f"  Loading: {builds_csv.name}")
        builds_df = pd.read_csv(builds_csv)

    return test_df, builds_df


def determine_test_result(row: pd.Series) -> str:
    """Determine if a test passed or failed based on available columns."""
    # Check common column names
    if 'failures' in row.index and 'errors' in row.index:
        if row['failures'] > 0 or row['errors'] > 0:
            return "Fail"
        return "Pass"

    if 'result' in row.index:
        result = str(row['result']).lower()
        if result in ['pass', 'passed', 'success', 'ok']:
            return "Pass"
        return "Fail"

    if 'status' in row.index:
        status = str(row['status']).lower()
        if status in ['pass', 'passed', 'success', 'ok']:
            return "Pass"
        return "Fail"

    if 'verdict' in row.index:
        verdict = str(row['verdict']).lower()
        if verdict in ['pass', 'passed', 'success', 'ok']:
            return "Pass"
        return "Fail"

    # Default to Pass if no failure indicators
    return "Pass"


def convert_project_to_filopriori_format(
    project_name: str,
    test_df: pd.DataFrame,
    builds_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Convert a project's test data to Filo-Priori format.

    Expected output columns:
    - Build_ID: Unique build identifier
    - TC_Key: Test case identifier
    - TE_Summary: Test description
    - TC_Steps: Test steps (empty for RTPTorrent)
    - TE_Test_Result: "Pass" or "Fail"
    - commit: Commit SHA
    - Build_Test_Start_Date: Timestamp
    """
    records = []

    # Determine column mappings based on available columns
    columns = test_df.columns.tolist()
    print(f"  Available columns: {columns}")

    # Build ID column
    build_col = None
    for col in ['travisJobId', 'travisBuildId', 'tr_build_id', 'build_id', 'buildId', 'build']:
        if col in columns:
            build_col = col
            break

    # Test name column
    test_col = None
    for col in ['testName', 'test_name', 'name', 'test_class', 'test']:
        if col in columns:
            test_col = col
            break

    if build_col is None or test_col is None:
        print(f"  Warning: Could not find required columns. Build: {build_col}, Test: {test_col}")
        return pd.DataFrame()

    # Process each row
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"  Processing {project_name}"):
        build_id = f"{project_name}_{row[build_col]}"
        test_name = str(row[test_col])

        # Parse test name
        package, class_name, method = parse_test_name(test_name)

        # Generate test key and description
        tc_key = f"{project_name}::{class_name}::{method}" if method else f"{project_name}::{class_name}"
        te_summary = generate_test_description(package, class_name, method)

        # Determine result
        te_result = determine_test_result(row)

        # Get commit if available
        commit = ""
        if builds_df is not None and 'git_all_built_commits' in builds_df.columns:
            build_info = builds_df[builds_df['tr_build_id'] == row[build_col]]
            if not build_info.empty:
                commits = str(build_info.iloc[0]['git_all_built_commits'])
                commit = commits.split('#')[0] if '#' in commits else commits

        # Get timestamp if available
        timestamp = ""
        if 'gh_build_started_at' in test_df.columns:
            timestamp = str(row['gh_build_started_at'])
        elif builds_df is not None and 'gh_build_started_at' in builds_df.columns:
            build_info = builds_df[builds_df['tr_build_id'] == row[build_col]]
            if not build_info.empty:
                timestamp = str(build_info.iloc[0]['gh_build_started_at'])

        records.append({
            'Build_ID': build_id,
            'TC_Key': tc_key,
            'TE_Summary': te_summary,
            'TC_Steps': "",  # Not available in RTPTorrent
            'TE_Test_Result': te_result,
            'commit': commit,
            'Build_Test_Start_Date': timestamp,
            'project': project_name,  # Extra field for filtering
        })

    return pd.DataFrame(records)


def split_train_test(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets by build (temporal split).

    Args:
        df: Full dataset
        test_ratio: Proportion of builds for test set

    Returns:
        (train_df, test_df)
    """
    # Get unique builds sorted by timestamp (if available) or name
    builds = df['Build_ID'].unique()

    # Simple split by build order (assuming chronological)
    n_test = int(len(builds) * test_ratio)
    test_builds = set(builds[-n_test:])
    train_builds = set(builds[:-n_test])

    train_df = df[df['Build_ID'].isin(train_builds)].copy()
    test_df = df[df['Build_ID'].isin(test_builds)].copy()

    return train_df, test_df


def compute_statistics(df: pd.DataFrame, name: str) -> Dict:
    """Compute dataset statistics."""
    stats = {
        'name': name,
        'total_executions': len(df),
        'unique_builds': df['Build_ID'].nunique(),
        'unique_test_cases': df['TC_Key'].nunique(),
        'pass_count': len(df[df['TE_Test_Result'] == 'Pass']),
        'fail_count': len(df[df['TE_Test_Result'] == 'Fail']),
    }

    stats['pass_fail_ratio'] = stats['pass_count'] / max(stats['fail_count'], 1)

    # Builds with failures
    builds_with_failures = df[df['TE_Test_Result'] == 'Fail']['Build_ID'].nunique()
    stats['builds_with_failures'] = builds_with_failures
    stats['failure_rate'] = builds_with_failures / max(stats['unique_builds'], 1)

    if 'project' in df.columns:
        stats['projects'] = df['project'].nunique()
        stats['projects_list'] = df['project'].unique().tolist()

    return stats


def main():
    """Main preprocessing routine."""
    parser = argparse.ArgumentParser(description="Preprocess RTPTorrent dataset")
    parser.add_argument('--project', type=str, help='Process only specific project')
    parser.add_argument('--sample', type=int, help='Create sample with N builds per project')
    parser.add_argument('--list-projects', action='store_true', help='List available projects')
    args = parser.parse_args()

    print("=" * 60)
    print("RTPTorrent Preprocessor for Filo-Priori")
    print("=" * 60)

    # Find data directory
    data_dir = find_rtptorrent_data(RAW_DIR)
    if data_dir is None:
        print(f"\nError: RTPTorrent data not found in {RAW_DIR}")
        print("Please run the download script first:")
        print("  python scripts/preprocessing/download_rtptorrent.py")
        sys.exit(1)

    print(f"\nData directory: {data_dir}")

    # List projects
    projects = list_available_projects(data_dir)
    if not projects:
        print("Error: No projects found in data directory")
        sys.exit(1)

    print(f"Found {len(projects)} projects: {', '.join(projects[:5])}...")

    if args.list_projects:
        print("\nAvailable projects:")
        for p in projects:
            print(f"  - {p}")
        return

    # Filter to specific project if requested
    if args.project:
        if args.project in projects:
            projects = [args.project]
        else:
            print(f"Error: Project '{args.project}' not found")
            print(f"Available: {projects}")
            sys.exit(1)

    # Process each project
    all_data = []

    for project_name in projects:
        print(f"\n{'='*60}")
        print(f"Processing: {project_name}")
        print("=" * 60)

        project_dir = data_dir / project_name

        try:
            test_df, builds_df = load_project_data(project_dir)
            converted_df = convert_project_to_filopriori_format(
                project_name, test_df, builds_df
            )

            if len(converted_df) > 0:
                all_data.append(converted_df)
                print(f"  Converted {len(converted_df)} test executions")
            else:
                print(f"  Warning: No data converted for {project_name}")

        except Exception as e:
            print(f"  Error processing {project_name}: {e}")
            continue

    if not all_data:
        print("\nError: No data was converted")
        sys.exit(1)

    # Combine all data
    print("\n" + "=" * 60)
    print("Combining all projects...")
    full_df = pd.concat(all_data, ignore_index=True)

    # Sample if requested
    if args.sample:
        print(f"Creating sample with {args.sample} builds per project...")
        sampled = []
        for project in full_df['project'].unique():
            project_df = full_df[full_df['project'] == project]
            builds = project_df['Build_ID'].unique()[:args.sample]
            sampled.append(project_df[project_df['Build_ID'].isin(builds)])
        full_df = pd.concat(sampled, ignore_index=True)

    # Split into train/test
    print("Splitting into train/test sets...")
    train_df, test_df = split_train_test(full_df, test_ratio=0.2)

    # Remove auxiliary columns
    columns_to_keep = [
        'Build_ID', 'TC_Key', 'TE_Summary', 'TC_Steps',
        'TE_Test_Result', 'commit', 'Build_Test_Start_Date'
    ]
    train_df = train_df[columns_to_keep]
    test_df = test_df[columns_to_keep]

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save datasets
    print(f"\nSaving to {OUTPUT_DIR}...")
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    # Compute and save statistics
    train_stats = compute_statistics(train_df, "train")
    test_stats = compute_statistics(test_df, "test")
    full_stats = compute_statistics(full_df, "full")

    stats = {
        'train': train_stats,
        'test': test_stats,
        'full': full_stats,
        'preprocessing_date': datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)

    print("\nTrain Set Statistics:")
    print(f"  Total executions: {train_stats['total_executions']:,}")
    print(f"  Unique builds: {train_stats['unique_builds']:,}")
    print(f"  Unique test cases: {train_stats['unique_test_cases']:,}")
    print(f"  Pass:Fail ratio: {train_stats['pass_fail_ratio']:.1f}:1")
    print(f"  Builds with failures: {train_stats['builds_with_failures']} ({train_stats['failure_rate']:.1%})")

    print("\nTest Set Statistics:")
    print(f"  Total executions: {test_stats['total_executions']:,}")
    print(f"  Unique builds: {test_stats['unique_builds']:,}")
    print(f"  Unique test cases: {test_stats['unique_test_cases']:,}")
    print(f"  Pass:Fail ratio: {test_stats['pass_fail_ratio']:.1f}:1")
    print(f"  Builds with failures: {test_stats['builds_with_failures']} ({test_stats['failure_rate']:.1%})")

    print("\nOutput files:")
    print(f"  {OUTPUT_DIR / 'train.csv'}")
    print(f"  {OUTPUT_DIR / 'test.csv'}")
    print(f"  {OUTPUT_DIR / 'statistics.json'}")

    print("\n" + "=" * 60)
    print("Next step: Run experiments with the new dataset:")
    print("  python main.py --config configs/experiment_rtptorrent.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
