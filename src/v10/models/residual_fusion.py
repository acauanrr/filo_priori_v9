"""
Residual Fusion Module for Filo-Priori V10.

This module implements the key innovation of V10: Residual Learning
with heuristic bias. Instead of learning from scratch, the neural
network learns to CORRECT the strong heuristic baseline.

Concept:
    score_final = α × score_heuristic + (1-α) × δ_neural

    Where:
    - score_heuristic: Score from Recently-Failed and other heuristics
    - δ_neural: Learned correction/residual from neural network
    - α: Learnable mixing weight (starts high, trusting heuristics)

This approach:
    1. Never forgets the strong heuristic signal
    2. Allows neural network to focus on edge cases
    3. Provides interpretability (we can inspect α)
    4. Improves stability (fallback to heuristic if neural fails)
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualFusion(nn.Module):
    """
    Simple linear residual fusion.

    score = α × h + (1 - α) × δ

    Where α is a learnable scalar initialized to favor heuristics.

    Args:
        initial_alpha: Initial value for α (0-1). Default 0.7 trusts heuristics.
        learnable_alpha: Whether α should be learned or fixed.
        temperature: Softmax temperature for α (if using sigmoid).

    Example:
        >>> fusion = ResidualFusion(initial_alpha=0.7)
        >>> h_heuristic = torch.tensor([0.9, 0.1, 0.5])  # Heuristic scores
        >>> delta_neural = torch.tensor([0.1, 0.3, -0.2])  # Neural corrections
        >>> scores = fusion(h_heuristic, delta_neural)
    """

    def __init__(
        self,
        initial_alpha: float = 0.7,
        learnable_alpha: bool = True,
        temperature: float = 1.0
    ):
        super().__init__()

        self.temperature = temperature
        self.learnable_alpha = learnable_alpha

        # Initialize α in logit space for numerical stability
        # α = sigmoid(alpha_logit)
        initial_logit = self._inverse_sigmoid(initial_alpha)

        if learnable_alpha:
            self.alpha_logit = nn.Parameter(torch.tensor(initial_logit))
        else:
            self.register_buffer('alpha_logit', torch.tensor(initial_logit))

    def _inverse_sigmoid(self, x: float) -> float:
        """Compute inverse sigmoid (logit)."""
        x = max(min(x, 0.999), 0.001)  # Clamp to avoid inf
        return torch.log(torch.tensor(x / (1 - x))).item()

    def forward(
        self,
        h_heuristic: torch.Tensor,
        delta_neural: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse heuristic score with neural correction.

        Args:
            h_heuristic: Heuristic-based scores [batch] or [batch, 1].
            delta_neural: Neural network predictions [batch] or [batch, 1].

        Returns:
            Fused scores [batch].
        """
        # Ensure same shape
        h_heuristic = h_heuristic.squeeze(-1) if h_heuristic.dim() > 1 else h_heuristic
        delta_neural = delta_neural.squeeze(-1) if delta_neural.dim() > 1 else delta_neural

        # Compute α
        alpha = torch.sigmoid(self.alpha_logit / self.temperature)

        # Linear combination
        score = alpha * h_heuristic + (1 - alpha) * delta_neural

        return score

    def get_alpha(self) -> float:
        """Get current α value."""
        return torch.sigmoid(self.alpha_logit).item()


class GatedResidualFusion(nn.Module):
    """
    Gated residual fusion with input-dependent α.

    Instead of a fixed α, this module learns to predict α
    based on the input features. This allows the model to
    trust heuristics more for some inputs and neural network
    more for others.

    Insight:
        - For tests with rich failure history: trust heuristics (high α)
        - For new tests or edge cases: trust neural network (low α)

    Args:
        heuristic_dim: Dimension of heuristic features.
        neural_dim: Dimension of neural features.
        hidden_dim: Hidden dimension for gate network.
        initial_bias: Bias towards heuristics (positive = trust heuristics).
    """

    def __init__(
        self,
        heuristic_dim: int = 6,
        neural_dim: int = 256,
        hidden_dim: int = 64,
        initial_bias: float = 1.0
    ):
        super().__init__()

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(heuristic_dim + neural_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize with bias towards heuristics
        with torch.no_grad():
            self.gate[-1].bias.fill_(initial_bias)

        # Heuristic score predictor
        self.heuristic_scorer = nn.Linear(heuristic_dim, 1)

        # Neural scorer
        self.neural_scorer = nn.Linear(neural_dim, 1)

    def forward(
        self,
        h_heuristic: torch.Tensor,
        h_neural: torch.Tensor,
        return_alpha: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fuse features with input-dependent gating.

        Args:
            h_heuristic: Heuristic features [batch, heuristic_dim].
            h_neural: Neural features [batch, neural_dim].
            return_alpha: Whether to return α values for analysis.

        Returns:
            scores: Fused scores [batch].
            alpha: Optional α values [batch] if return_alpha=True.
        """
        # Compute input-dependent α
        combined = torch.cat([h_heuristic, h_neural], dim=-1)
        alpha_logit = self.gate(combined)
        alpha = torch.sigmoid(alpha_logit)  # [batch, 1]

        # Compute scores from each branch
        score_heuristic = self.heuristic_scorer(h_heuristic)  # [batch, 1]
        score_neural = self.neural_scorer(h_neural)  # [batch, 1]

        # Gated fusion
        score = alpha * score_heuristic + (1 - alpha) * score_neural
        score = score.squeeze(-1)

        if return_alpha:
            return score, alpha.squeeze(-1)
        return score, None


class HierarchicalResidualFusion(nn.Module):
    """
    Hierarchical residual fusion for multiple feature sources.

    V10 has three main signal sources:
    1. Heuristics (recency, fail_rate, etc.)
    2. Semantic (CodeBERT embeddings)
    3. Structural (Graph embeddings)

    This module fuses them hierarchically:
    - First: Semantic + Structural → Neural
    - Then: Neural + Heuristics → Final

    This ensures heuristics always have a direct path to output.
    """

    def __init__(
        self,
        heuristic_dim: int = 6,
        semantic_dim: int = 768,
        structural_dim: int = 256,
        hidden_dim: int = 256,
        initial_heuristic_weight: float = 0.5
    ):
        super().__init__()

        # Stage 1: Fuse semantic + structural
        self.neural_fusion = nn.Sequential(
            nn.Linear(semantic_dim + structural_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Stage 2: Fuse neural + heuristics
        self.final_fusion = GatedResidualFusion(
            heuristic_dim=heuristic_dim,
            neural_dim=hidden_dim,
            hidden_dim=64,
            initial_bias=self._compute_initial_bias(initial_heuristic_weight)
        )

    def _compute_initial_bias(self, weight: float) -> float:
        """Convert desired weight to logit bias."""
        weight = max(min(weight, 0.99), 0.01)
        return torch.log(torch.tensor(weight / (1 - weight))).item()

    def forward(
        self,
        h_heuristic: torch.Tensor,
        h_semantic: torch.Tensor,
        h_structural: torch.Tensor,
        return_intermediates: bool = False
    ):
        """
        Hierarchical fusion of all feature sources.

        Args:
            h_heuristic: Heuristic features [batch, heuristic_dim].
            h_semantic: Semantic (CodeBERT) features [batch, semantic_dim].
            h_structural: Structural (Graph) features [batch, structural_dim].
            return_intermediates: Return intermediate representations.

        Returns:
            scores: Final scores [batch].
            intermediates: Optional dict of intermediate values.
        """
        # Stage 1: Neural fusion
        h_combined = torch.cat([h_semantic, h_structural], dim=-1)
        h_neural = self.neural_fusion(h_combined)

        # Stage 2: Residual fusion with heuristics
        scores, alpha = self.final_fusion(h_heuristic, h_neural, return_alpha=True)

        if return_intermediates:
            return scores, {
                'h_neural': h_neural,
                'alpha': alpha
            }

        return scores, None


class AttentiveResidualFusion(nn.Module):
    """
    Attention-based residual fusion.

    Uses attention to dynamically weight different feature sources.
    More powerful than simple gating but also more complex.
    """

    def __init__(
        self,
        feature_dims: dict,
        num_heads: int = 4,
        hidden_dim: int = 256
    ):
        """
        Args:
            feature_dims: Dict mapping feature name to dimension.
                         e.g., {'heuristic': 6, 'semantic': 768, 'structural': 256}
            num_heads: Number of attention heads.
            hidden_dim: Common hidden dimension for all features.
        """
        super().__init__()

        self.feature_names = list(feature_dims.keys())

        # Project all features to common dimension
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim)
            for name, dim in feature_dims.items()
        })

        # Multi-head attention for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Output scorer
        self.scorer = nn.Linear(hidden_dim, 1)

        # Learnable query for aggregation
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, features: dict) -> torch.Tensor:
        """
        Fuse features using attention.

        Args:
            features: Dict mapping feature name to tensor.
                     Each tensor should be [batch, feature_dim].

        Returns:
            scores: [batch]
        """
        batch_size = list(features.values())[0].shape[0]

        # Project all features
        projected = []
        for name in self.feature_names:
            if name in features:
                h = self.projections[name](features[name])  # [batch, hidden]
                projected.append(h)

        # Stack as sequence: [batch, num_features, hidden]
        keys = torch.stack(projected, dim=1)
        values = keys

        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)  # [batch, 1, hidden]

        # Attention
        attended, weights = self.attention(query, keys, values)  # [batch, 1, hidden]

        # Score
        scores = self.scorer(attended.squeeze(1))  # [batch, 1]

        return scores.squeeze(-1)
