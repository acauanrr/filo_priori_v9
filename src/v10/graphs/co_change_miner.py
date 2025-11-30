"""
Co-Change Miner for Git Repository Analysis.

This module extracts co-change patterns from Git history to build
the temporal graph for Filo-Priori V10.

It parses Git logs to identify which files frequently change together,
providing the raw data for TimeDecayGraphBuilder.
"""

import re
import logging
import subprocess
from datetime import datetime
from typing import List, Dict, Optional, Set, Generator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CommitInfo:
    """Represents a single Git commit."""
    hash: str
    timestamp: datetime
    author: str
    message: str
    changed_files: List[str]


class CoChangeMiner:
    """
    Mines co-change patterns from Git repository history.

    This class parses Git logs to extract:
    1. Which files changed together in each commit
    2. When these changes occurred
    3. File-level statistics

    Args:
        repo_path: Path to the Git repository root.
        file_extensions: Only include files with these extensions.
        exclude_patterns: Regex patterns for files to exclude.
        max_files_per_commit: Skip commits with too many files (likely merges).

    Example:
        >>> miner = CoChangeMiner('/path/to/repo', file_extensions=['.java'])
        >>> commits = miner.mine_commits(since='2023-01-01')
        >>> for commit in commits:
        ...     print(f"{commit.hash}: {len(commit.changed_files)} files")
    """

    def __init__(
        self,
        repo_path: str,
        file_extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_files_per_commit: int = 50
    ):
        self.repo_path = Path(repo_path)
        self.file_extensions = file_extensions or ['.java', '.py', '.js', '.ts']
        self.exclude_patterns = [
            re.compile(p) for p in (exclude_patterns or [
                r'test.*fixture',
                r'\.min\.',
                r'vendor/',
                r'node_modules/',
                r'__pycache__/',
            ])
        ]
        self.max_files_per_commit = max_files_per_commit

    def mine_commits(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
        branch: str = 'HEAD',
        max_commits: Optional[int] = None
    ) -> Generator[CommitInfo, None, None]:
        """
        Mine commits from Git history.

        Args:
            since: Only commits after this date (YYYY-MM-DD).
            until: Only commits before this date.
            branch: Branch to analyze.
            max_commits: Maximum number of commits to process.

        Yields:
            CommitInfo objects for each qualifying commit.
        """
        # Build git log command
        cmd = [
            'git', '-C', str(self.repo_path),
            'log', branch,
            '--name-only',
            '--pretty=format:%H|%aI|%an|%s',
            '--no-merges'  # Skip merge commits
        ]

        if since:
            cmd.append(f'--since={since}')
        if until:
            cmd.append(f'--until={until}')
        if max_commits:
            cmd.append(f'-n{max_commits}')

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e.stderr}")
            return

        # Parse output
        current_commit = None
        current_files = []

        for line in result.stdout.split('\n'):
            line = line.strip()

            if not line:
                # Empty line separates commits
                if current_commit and current_files:
                    filtered_files = self._filter_files(current_files)
                    if filtered_files and len(filtered_files) <= self.max_files_per_commit:
                        yield CommitInfo(
                            hash=current_commit['hash'],
                            timestamp=current_commit['timestamp'],
                            author=current_commit['author'],
                            message=current_commit['message'],
                            changed_files=filtered_files
                        )
                current_commit = None
                current_files = []
                continue

            if '|' in line and current_commit is None:
                # Parse commit header
                parts = line.split('|', 3)
                if len(parts) >= 4:
                    try:
                        timestamp = datetime.fromisoformat(parts[1].replace('Z', '+00:00'))
                    except ValueError:
                        timestamp = datetime.now()

                    current_commit = {
                        'hash': parts[0],
                        'timestamp': timestamp,
                        'author': parts[2],
                        'message': parts[3]
                    }
            elif current_commit is not None:
                # This is a file path
                current_files.append(line)

        # Don't forget the last commit
        if current_commit and current_files:
            filtered_files = self._filter_files(current_files)
            if filtered_files and len(filtered_files) <= self.max_files_per_commit:
                yield CommitInfo(
                    hash=current_commit['hash'],
                    timestamp=current_commit['timestamp'],
                    author=current_commit['author'],
                    message=current_commit['message'],
                    changed_files=filtered_files
                )

    def _filter_files(self, files: List[str]) -> List[str]:
        """Filter files by extension and exclude patterns."""
        filtered = []

        for f in files:
            # Check extension
            if self.file_extensions:
                if not any(f.endswith(ext) for ext in self.file_extensions):
                    continue

            # Check exclude patterns
            if any(pattern.search(f) for pattern in self.exclude_patterns):
                continue

            filtered.append(f)

        return filtered

    def mine_to_dict(self, **kwargs) -> List[Dict]:
        """
        Mine commits and return as list of dicts.

        Convenient format for TimeDecayGraphBuilder.build_from_git_log().
        """
        return [
            {
                'files': commit.changed_files,
                'timestamp': commit.timestamp,
                'hash': commit.hash
            }
            for commit in self.mine_commits(**kwargs)
        ]

    def get_file_statistics(self, **kwargs) -> Dict[str, Dict]:
        """
        Compute per-file statistics from commit history.

        Returns dict with:
        - change_count: Number of commits touching the file
        - first_change: First commit timestamp
        - last_change: Last commit timestamp
        - co_change_partners: Files it commonly changes with
        """
        from collections import defaultdict

        stats = defaultdict(lambda: {
            'change_count': 0,
            'first_change': None,
            'last_change': None,
            'co_change_partners': defaultdict(int)
        })

        for commit in self.mine_commits(**kwargs):
            for f in commit.changed_files:
                stats[f]['change_count'] += 1

                if stats[f]['first_change'] is None:
                    stats[f]['first_change'] = commit.timestamp
                stats[f]['last_change'] = commit.timestamp

                # Track co-change partners
                for other in commit.changed_files:
                    if other != f:
                        stats[f]['co_change_partners'][other] += 1

        # Convert defaultdicts to regular dicts
        return {
            f: {
                'change_count': s['change_count'],
                'first_change': s['first_change'],
                'last_change': s['last_change'],
                'co_change_partners': dict(s['co_change_partners'])
            }
            for f, s in stats.items()
        }

    def get_test_production_mapping(
        self,
        test_patterns: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Set[str]]:
        """
        Build a mapping from test files to production files they co-change with.

        This is useful for identifying which tests are likely affected by
        production code changes.

        Args:
            test_patterns: Regex patterns to identify test files.
                          Default: files containing 'Test' or 'test'.

        Returns:
            Dict mapping test file -> set of production files
        """
        test_patterns = test_patterns or [r'[Tt]est', r'[Ss]pec']
        test_regexes = [re.compile(p) for p in test_patterns]

        def is_test_file(f: str) -> bool:
            return any(r.search(f) for r in test_regexes)

        mapping = {}

        for commit in self.mine_commits(**kwargs):
            test_files = [f for f in commit.changed_files if is_test_file(f)]
            prod_files = [f for f in commit.changed_files if not is_test_file(f)]

            for test in test_files:
                if test not in mapping:
                    mapping[test] = set()
                mapping[test].update(prod_files)

        return mapping


class RTPTorrentCoChangeMiner:
    """
    Specialized miner for RTPTorrent dataset.

    RTPTorrent provides pre-processed build logs, not raw Git history.
    This class adapts the co-change mining for RTPTorrent's format.
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def mine_from_builds(
        self,
        project_name: str
    ) -> Generator[Dict, None, None]:
        """
        Extract co-change events from RTPTorrent build logs.

        In RTPTorrent, we infer co-changes from:
        1. Tests that fail together in the same build
        2. Tests executed in the same build (weaker signal)
        """
        import pandas as pd

        project_dir = self.dataset_path / project_name
        if not project_dir.exists():
            logger.error(f"Project not found: {project_name}")
            return

        # Load build history
        # RTPTorrent format: build_id, test_name, verdict, duration, etc.
        builds_file = project_dir / 'builds.csv'
        if not builds_file.exists():
            logger.error(f"Builds file not found: {builds_file}")
            return

        df = pd.read_csv(builds_file)

        # Group by build
        for build_id, group in df.groupby('build_id'):
            # Get timestamp (if available)
            timestamp = group['timestamp'].iloc[0] if 'timestamp' in group.columns else datetime.now()
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.now()

            # Get test names
            tests = group['test_name'].tolist()

            yield {
                'files': tests,  # Treat tests as "files"
                'timestamp': timestamp,
                'hash': str(build_id)
            }

    def mine_failure_co_occurrences(
        self,
        project_name: str
    ) -> Generator[Dict, None, None]:
        """
        Extract co-failure events (tests that fail together).

        This provides a stronger signal than general co-execution.
        """
        import pandas as pd

        project_dir = self.dataset_path / project_name
        builds_file = project_dir / 'builds.csv'

        if not builds_file.exists():
            return

        df = pd.read_csv(builds_file)

        # Only look at failures
        failures = df[df['verdict'] == 'FAIL']

        for build_id, group in failures.groupby('build_id'):
            timestamp = group['timestamp'].iloc[0] if 'timestamp' in group.columns else datetime.now()
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.now()

            failing_tests = group['test_name'].tolist()

            if len(failing_tests) >= 2:
                yield {
                    'files': failing_tests,
                    'timestamp': timestamp,
                    'hash': f"fail_{build_id}"
                }
