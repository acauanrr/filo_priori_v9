"""
Time-Decay Graph Builder for Filo-Priori V10.

This module implements a dynamic graph where edge weights decay exponentially
with time, capturing the "burstiness" of software failures.

Key Innovation:
    Unlike V9 (static weights: 1.0, 0.5, 0.3), V10 uses:
    W_ij(t) = Σ exp(-λ * (t - t_k))

    This ensures recent co-changes have much more influence than old ones.

References:
    - Temporal locality in software failures
    - Burstiness patterns in CI/CD pipelines
"""

import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


@dataclass
class CoChangeEvent:
    """Represents a co-change event between two files."""
    file_a: str
    file_b: str
    timestamp: datetime
    commit_hash: str


class TimeDecayGraphBuilder:
    """
    Builds a co-change graph with exponential time decay on edge weights.

    The graph captures how files change together over time, with recent
    co-changes weighted more heavily than old ones.

    Mathematical Formulation:
        W_ij(t) = Σ_{k ∈ Commits} I(co_change_k(i,j)) × exp(-λ × (t - t_k))

    Where:
        - t = current time (query time)
        - t_k = timestamp of commit k
        - λ = decay rate (higher = faster decay)
        - I() = indicator function (1 if files changed together)

    Args:
        decay_lambda: Decay rate parameter. Default 0.1 means ~90% weight
                     remains after 1 day, ~37% after 10 days.
        lookback_days: How far back to look for co-changes.
        min_co_changes: Minimum co-changes to create an edge.
        normalize_weights: Whether to normalize weights to [0, 1].

    Example:
        >>> builder = TimeDecayGraphBuilder(decay_lambda=0.1)
        >>> graph = builder.build_from_history(co_change_events, query_time)
        >>> print(graph.edge_index.shape)  # [2, num_edges]
        >>> print(graph.edge_weight.shape)  # [num_edges]
    """

    def __init__(
        self,
        decay_lambda: float = 0.1,
        lookback_days: int = 365,
        min_co_changes: int = 2,
        normalize_weights: bool = True,
        include_self_loops: bool = False
    ):
        self.decay_lambda = decay_lambda
        self.lookback_days = lookback_days
        self.min_co_changes = min_co_changes
        self.normalize_weights = normalize_weights
        self.include_self_loops = include_self_loops

        # Internal state
        self._file_to_idx: Dict[str, int] = {}
        self._idx_to_file: Dict[int, str] = {}
        self._co_change_history: Dict[Tuple[str, str], List[datetime]] = defaultdict(list)

    def compute_decay_weight(
        self,
        co_change_times: List[datetime],
        query_time: datetime
    ) -> float:
        """
        Compute the decayed weight for a set of co-change events.

        Formula: W = Σ exp(-λ × Δt)

        Args:
            co_change_times: List of timestamps when files changed together.
            query_time: Current time to compute decay from.

        Returns:
            Aggregated weight with exponential decay applied.

        Example:
            >>> times = [datetime.now() - timedelta(days=1)]
            >>> weight = builder.compute_decay_weight(times, datetime.now())
            >>> print(f"Weight after 1 day: {weight:.4f}")  # ~0.9048
        """
        if not co_change_times:
            return 0.0

        total_weight = 0.0
        cutoff_time = query_time - timedelta(days=self.lookback_days)

        for t_k in co_change_times:
            # Skip events outside lookback window
            if t_k < cutoff_time:
                continue

            # Compute time delta in days (fractional)
            delta_days = (query_time - t_k).total_seconds() / 86400.0

            # Exponential decay
            weight = math.exp(-self.decay_lambda * delta_days)
            total_weight += weight

        return total_weight

    def add_co_change_event(self, event: CoChangeEvent) -> None:
        """
        Add a single co-change event to the history.

        Args:
            event: CoChangeEvent with file pair and timestamp.
        """
        # Ensure consistent ordering (alphabetical)
        file_a, file_b = sorted([event.file_a, event.file_b])
        key = (file_a, file_b)

        self._co_change_history[key].append(event.timestamp)

        # Update file indices
        for f in [file_a, file_b]:
            if f not in self._file_to_idx:
                idx = len(self._file_to_idx)
                self._file_to_idx[f] = idx
                self._idx_to_file[idx] = f

    def add_commit(
        self,
        changed_files: List[str],
        timestamp: datetime,
        commit_hash: str = ""
    ) -> None:
        """
        Add all co-change pairs from a single commit.

        Args:
            changed_files: List of files changed in the commit.
            timestamp: When the commit was made.
            commit_hash: Optional commit identifier.
        """
        # Generate all pairs
        for i, file_a in enumerate(changed_files):
            for file_b in changed_files[i + 1:]:
                event = CoChangeEvent(
                    file_a=file_a,
                    file_b=file_b,
                    timestamp=timestamp,
                    commit_hash=commit_hash
                )
                self.add_co_change_event(event)

    def build_graph(
        self,
        query_time: Optional[datetime] = None,
        target_files: Optional[Set[str]] = None
    ) -> Data:
        """
        Build the PyTorch Geometric graph with time-decayed edge weights.

        Args:
            query_time: Time to compute decay from. Defaults to now.
            target_files: Optional subset of files to include. If None,
                         includes all files in history.

        Returns:
            PyG Data object with:
                - x: Node features (placeholder, identity)
                - edge_index: [2, num_edges] edge connections
                - edge_weight: [num_edges] decayed weights
                - file_mapping: Dict mapping file names to indices
        """
        if query_time is None:
            query_time = datetime.now()

        # Determine which files to include
        if target_files is not None:
            files = [f for f in target_files if f in self._file_to_idx]
        else:
            files = list(self._file_to_idx.keys())

        if not files:
            logger.warning("No files found for graph construction")
            return self._empty_graph()

        # Create local file indexing
        local_file_to_idx = {f: i for i, f in enumerate(files)}
        num_nodes = len(files)

        # Compute edge weights
        edges_src = []
        edges_dst = []
        edge_weights = []

        for (file_a, file_b), timestamps in self._co_change_history.items():
            # Skip if files not in target set
            if file_a not in local_file_to_idx or file_b not in local_file_to_idx:
                continue

            # Skip if below minimum co-changes
            if len(timestamps) < self.min_co_changes:
                continue

            # Compute decayed weight
            weight = self.compute_decay_weight(timestamps, query_time)

            if weight > 0:
                idx_a = local_file_to_idx[file_a]
                idx_b = local_file_to_idx[file_b]

                # Add bidirectional edges (undirected graph)
                edges_src.extend([idx_a, idx_b])
                edges_dst.extend([idx_b, idx_a])
                edge_weights.extend([weight, weight])

        # Add self-loops if requested
        if self.include_self_loops:
            for i in range(num_nodes):
                edges_src.append(i)
                edges_dst.append(i)
                edge_weights.append(1.0)

        # Convert to tensors
        if edges_src:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            edge_weight = torch.tensor(edge_weights, dtype=torch.float32)

            # Normalize weights to [0, 1]
            if self.normalize_weights and edge_weight.numel() > 0:
                max_weight = edge_weight.max()
                if max_weight > 0:
                    edge_weight = edge_weight / max_weight
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weight = torch.zeros(0, dtype=torch.float32)

        # Node features (placeholder - will be replaced by embeddings)
        x = torch.eye(num_nodes, dtype=torch.float32)

        # Create PyG Data object
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=num_nodes
        )

        # Store mapping for later use
        graph.file_mapping = local_file_to_idx
        graph.idx_to_file = {v: k for k, v in local_file_to_idx.items()}

        logger.info(
            f"Built time-decay graph: {num_nodes} nodes, "
            f"{edge_index.shape[1]} edges, "
            f"weight range [{edge_weight.min():.4f}, {edge_weight.max():.4f}]"
        )

        return graph

    def build_from_git_log(
        self,
        git_log_entries: List[Dict],
        query_time: Optional[datetime] = None
    ) -> Data:
        """
        Build graph directly from parsed Git log entries.

        Args:
            git_log_entries: List of dicts with 'files', 'timestamp', 'hash'.
            query_time: Time to compute decay from.

        Returns:
            PyG Data object.

        Example:
            >>> entries = [
            ...     {'files': ['A.java', 'B.java'], 'timestamp': datetime(...), 'hash': 'abc123'},
            ...     {'files': ['B.java', 'C.java'], 'timestamp': datetime(...), 'hash': 'def456'},
            ... ]
            >>> graph = builder.build_from_git_log(entries)
        """
        for entry in git_log_entries:
            self.add_commit(
                changed_files=entry['files'],
                timestamp=entry['timestamp'],
                commit_hash=entry.get('hash', '')
            )

        return self.build_graph(query_time=query_time)

    def get_edge_weight_at_time(
        self,
        file_a: str,
        file_b: str,
        query_time: datetime
    ) -> float:
        """
        Get the edge weight between two specific files at a given time.

        Useful for debugging and analysis.
        """
        key = tuple(sorted([file_a, file_b]))
        timestamps = self._co_change_history.get(key, [])
        return self.compute_decay_weight(timestamps, query_time)

    def get_statistics(self) -> Dict:
        """
        Get statistics about the co-change history.
        """
        num_pairs = len(self._co_change_history)
        total_events = sum(len(ts) for ts in self._co_change_history.values())

        co_change_counts = [len(ts) for ts in self._co_change_history.values()]

        return {
            'num_files': len(self._file_to_idx),
            'num_pairs': num_pairs,
            'total_co_change_events': total_events,
            'avg_co_changes_per_pair': total_events / max(num_pairs, 1),
            'max_co_changes': max(co_change_counts) if co_change_counts else 0,
            'min_co_changes': min(co_change_counts) if co_change_counts else 0,
        }

    def _empty_graph(self) -> Data:
        """Create an empty graph placeholder."""
        return Data(
            x=torch.zeros((0, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_weight=torch.zeros(0, dtype=torch.float32),
            num_nodes=0
        )

    def clear_history(self) -> None:
        """Clear all stored co-change history."""
        self._file_to_idx.clear()
        self._idx_to_file.clear()
        self._co_change_history.clear()

    def save(self, path: str) -> None:
        """Save the builder state to disk."""
        import pickle
        state = {
            'decay_lambda': self.decay_lambda,
            'lookback_days': self.lookback_days,
            'min_co_changes': self.min_co_changes,
            'normalize_weights': self.normalize_weights,
            'file_to_idx': self._file_to_idx,
            'co_change_history': dict(self._co_change_history),
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Saved TimeDecayGraphBuilder to {path}")

    @classmethod
    def load(cls, path: str) -> 'TimeDecayGraphBuilder':
        """Load a builder from disk."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)

        builder = cls(
            decay_lambda=state['decay_lambda'],
            lookback_days=state['lookback_days'],
            min_co_changes=state['min_co_changes'],
            normalize_weights=state['normalize_weights'],
        )
        builder._file_to_idx = state['file_to_idx']
        builder._idx_to_file = {v: k for k, v in state['file_to_idx'].items()}
        builder._co_change_history = defaultdict(list, state['co_change_history'])

        logger.info(f"Loaded TimeDecayGraphBuilder from {path}")
        return builder


class TimeDecayGraphBuilderV2(TimeDecayGraphBuilder):
    """
    Extended version with additional decay functions.

    Supports:
    - Exponential decay (default)
    - Linear decay
    - Logarithmic decay
    - Custom decay functions
    """

    def __init__(
        self,
        decay_type: str = 'exponential',
        decay_lambda: float = 0.1,
        **kwargs
    ):
        super().__init__(decay_lambda=decay_lambda, **kwargs)
        self.decay_type = decay_type

    def compute_decay_weight(
        self,
        co_change_times: List[datetime],
        query_time: datetime
    ) -> float:
        """Compute weight using the specified decay function."""
        if not co_change_times:
            return 0.0

        total_weight = 0.0
        cutoff_time = query_time - timedelta(days=self.lookback_days)

        for t_k in co_change_times:
            if t_k < cutoff_time:
                continue

            delta_days = (query_time - t_k).total_seconds() / 86400.0

            if self.decay_type == 'exponential':
                weight = math.exp(-self.decay_lambda * delta_days)
            elif self.decay_type == 'linear':
                weight = max(0, 1 - self.decay_lambda * delta_days)
            elif self.decay_type == 'logarithmic':
                weight = 1.0 / math.log(2 + self.decay_lambda * delta_days)
            elif self.decay_type == 'inverse':
                weight = 1.0 / (1 + self.decay_lambda * delta_days)
            else:
                weight = math.exp(-self.decay_lambda * delta_days)

            total_weight += weight

        return total_weight
