"""
V10 Ranking Modules.

Implements Learning-to-Rank losses for direct APFD optimization.
"""

from .lambda_rank import LambdaRankLoss, LambdaLoss
from .approx_ndcg import ApproxNDCGLoss, SoftNDCGLoss
from .ndcg_utils import compute_ndcg, compute_dcg, compute_ideal_dcg

__all__ = [
    "LambdaRankLoss",
    "LambdaLoss",
    "ApproxNDCGLoss",
    "SoftNDCGLoss",
    "compute_ndcg",
    "compute_dcg",
    "compute_ideal_dcg",
]
