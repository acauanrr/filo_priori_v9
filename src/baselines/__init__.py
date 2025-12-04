"""
TCP Baselines from Literature

This module provides implementations of state-of-the-art baselines
for Test Case Prioritization comparison.

Available baselines:
    - RETECS: Reinforcement Learning for TCP (Spieker et al., ISSTA 2017)
    - DeepOrder: Deep Learning for TCP (Chen et al., ICSME 2021)
    - NodeRank: Mutation-based Ensemble for TCP (Li et al., IEEE TSE 2024)
    - Heuristic baselines: Random, FailureRate, Recency, etc.
"""

from .retecs import RETECSAgent, run_retecs_experiment
from .deeporder import DeepOrderModel, run_deeporder_experiment
from .noderank import NodeRankModel, run_noderank_experiment, compare_with_baselines

__all__ = [
    'RETECSAgent',
    'run_retecs_experiment',
    'DeepOrderModel',
    'run_deeporder_experiment',
    'NodeRankModel',
    'run_noderank_experiment',
    'compare_with_baselines'
]
