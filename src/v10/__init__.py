"""
Filo-Priori V10 - Hybrid Neuro-Symbolic Architecture

This module implements the V10 architecture with:
1. CodeBERT + Co-Attention (semantic encoding)
2. Time-Decay Graph + GAT (temporal structure)
3. LambdaRank Loss (direct APFD optimization)
4. Residual Learning (heuristic bias)

Target: Surpass "Recently-Failed" baseline on RTPTorrent dataset.
"""

from .models.hybrid_model import FiloPrioriV10
from .graphs.time_decay_builder import TimeDecayGraphBuilder
from .encoders.codebert_encoder import CodeBERTEncoder
from .ranking.lambda_rank import LambdaRankLoss

__version__ = "10.0.0"
__all__ = [
    "FiloPrioriV10",
    "TimeDecayGraphBuilder",
    "CodeBERTEncoder",
    "LambdaRankLoss",
]
