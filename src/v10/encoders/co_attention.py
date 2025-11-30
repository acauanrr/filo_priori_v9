"""
Co-Attention Module for Filo-Priori V10.

This module implements co-attention between test and code embeddings,
allowing the model to focus on the most relevant parts of each.

Co-Attention Mechanism:
    Given:
        h_test = embedding of test identifier
        h_code = embedding of code change

    The co-attention allows h_test to "query" h_code:
        attention = softmax(Q_test @ K_code^T)
        h_attended = attention @ V_code

    This identifies which parts of the code change are most
    relevant to the test.

Reference:
    Lu et al., "Hierarchical Question-Image Co-Attention for
    Visual Question Answering", NeurIPS 2016.
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoAttention(nn.Module):
    """
    Bidirectional co-attention between two embedding sequences.

    This allows mutual attention:
    1. Test attends to code (what code parts matter for this test?)
    2. Code attends to test (what test aspects matter for this code?)

    Args:
        hidden_dim: Dimension of input embeddings.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        use_residual: Whether to add residual connections.
        use_layer_norm: Whether to use layer normalization.

    Example:
        >>> co_attn = CoAttention(hidden_dim=768, num_heads=8)
        >>> h_test = torch.randn(32, 768)  # [batch, hidden]
        >>> h_code = torch.randn(32, 768)
        >>> h_test_new, h_code_new, weights = co_attn(h_test, h_code)
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Test attending to Code
        self.test_to_code_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Code attending to Test
        self.code_to_test_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norms
        if use_layer_norm:
            self.norm_test = nn.LayerNorm(hidden_dim)
            self.norm_code = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_test: torch.Tensor,
        h_code: torch.Tensor,
        return_weights: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Apply bidirectional co-attention.

        Args:
            h_test: Test embeddings [batch, hidden_dim] or [batch, seq_len, hidden_dim]
            h_code: Code embeddings [batch, hidden_dim] or [batch, seq_len, hidden_dim]
            return_weights: Whether to return attention weights.

        Returns:
            h_test_new: Updated test embeddings.
            h_code_new: Updated code embeddings.
            weights: Optional tuple of (test_to_code_weights, code_to_test_weights).
        """
        # Handle both 2D and 3D inputs
        if h_test.dim() == 2:
            h_test = h_test.unsqueeze(1)  # [batch, 1, hidden]
        if h_code.dim() == 2:
            h_code = h_code.unsqueeze(1)  # [batch, 1, hidden]

        # Test attends to Code
        # Query: test, Key/Value: code
        h_test_attended, attn_test_to_code = self.test_to_code_attn(
            query=h_test,
            key=h_code,
            value=h_code
        )

        # Code attends to Test
        # Query: code, Key/Value: test
        h_code_attended, attn_code_to_test = self.code_to_test_attn(
            query=h_code,
            key=h_test,
            value=h_test
        )

        # Residual connections
        if self.use_residual:
            h_test_new = h_test + self.dropout(h_test_attended)
            h_code_new = h_code + self.dropout(h_code_attended)
        else:
            h_test_new = h_test_attended
            h_code_new = h_code_attended

        # Layer normalization
        if self.use_layer_norm:
            h_test_new = self.norm_test(h_test_new)
            h_code_new = self.norm_code(h_code_new)

        # Squeeze back if input was 2D
        if h_test_new.shape[1] == 1:
            h_test_new = h_test_new.squeeze(1)
        if h_code_new.shape[1] == 1:
            h_code_new = h_code_new.squeeze(1)

        if return_weights:
            return h_test_new, h_code_new, (attn_test_to_code, attn_code_to_test)
        else:
            return h_test_new, h_code_new, None


class MultiHeadCoAttention(nn.Module):
    """
    Multi-head co-attention with feed-forward networks.

    This is a more powerful version that includes:
    1. Multi-head co-attention (as above)
    2. Feed-forward network for each stream
    3. Multiple layers of co-attention

    Similar to a Transformer encoder but with cross-attention.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            CoAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        h_test: torch.Tensor,
        h_code: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multiple layers of co-attention.

        Args:
            h_test: Test embeddings [batch, hidden_dim].
            h_code: Code embeddings [batch, hidden_dim].

        Returns:
            Updated (h_test, h_code) tuple.
        """
        for layer in self.layers:
            h_test, h_code = layer(h_test, h_code)

        return h_test, h_code


class CoAttentionLayer(nn.Module):
    """
    Single layer of co-attention with FFN.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        # Co-attention
        self.co_attn = CoAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Feed-forward networks
        self.ff_test = FeedForward(hidden_dim, ff_dim, dropout)
        self.ff_code = FeedForward(hidden_dim, ff_dim, dropout)

        # Layer norms
        self.norm_test = nn.LayerNorm(hidden_dim)
        self.norm_code = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_test: torch.Tensor,
        h_code: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply co-attention + FFN."""
        # Co-attention
        h_test_attn, h_code_attn, _ = self.co_attn(h_test, h_code, return_weights=False)

        # FFN for test
        h_test_out = self.norm_test(h_test_attn + self.ff_test(h_test_attn))

        # FFN for code
        h_code_out = self.norm_code(h_code_attn + self.ff_code(h_code_attn))

        return h_test_out, h_code_out


class FeedForward(nn.Module):
    """Standard feed-forward network."""

    def __init__(self, hidden_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GuidedAttention(nn.Module):
    """
    Guided attention that uses test embedding to guide code attention.

    This is a unidirectional version where only the test "guides"
    attention over the code, useful when test context should dominate.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()

        # Linear projections
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self.scale = math.sqrt(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_test: torch.Tensor,
        h_code: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply guided attention.

        Args:
            h_test: Guide embeddings [batch, hidden_dim].
            h_code: Target embeddings [batch, hidden_dim].

        Returns:
            attended: Attended code embeddings.
            weights: Attention weights.
        """
        # Ensure 3D for attention
        if h_test.dim() == 2:
            h_test = h_test.unsqueeze(1)
        if h_code.dim() == 2:
            h_code = h_code.unsqueeze(1)

        # Projections
        Q = self.query_proj(h_test)  # [batch, 1, hidden]
        K = self.key_proj(h_code)    # [batch, 1, hidden]
        V = self.value_proj(h_code)  # [batch, 1, hidden]

        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(-2, -1)) / self.scale  # [batch, 1, 1]
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Apply attention
        attended = torch.bmm(weights, V)  # [batch, 1, hidden]
        attended = self.output_proj(attended)

        # Residual + norm
        output = self.norm(h_code + attended)

        return output.squeeze(1), weights.squeeze(1)


class HierarchicalCoAttention(nn.Module):
    """
    Hierarchical co-attention for different granularity levels.

    Useful when you have:
    - Token-level embeddings (fine-grained)
    - Sentence/chunk-level embeddings (coarse-grained)

    Applies co-attention at each level and combines.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        # Token-level co-attention
        self.token_coattn = CoAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Sentence-level co-attention
        self.sentence_coattn = CoAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(
        self,
        h_test_token: torch.Tensor,
        h_code_token: torch.Tensor,
        h_test_sent: torch.Tensor,
        h_code_sent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply hierarchical co-attention.

        Args:
            h_test_token: Token-level test embeddings [batch, seq, hidden].
            h_code_token: Token-level code embeddings [batch, seq, hidden].
            h_test_sent: Sentence-level test embeddings [batch, hidden].
            h_code_sent: Sentence-level code embeddings [batch, hidden].

        Returns:
            h_test_final: Final test embedding.
            h_code_final: Final code embedding.
        """
        # Token-level
        h_test_tok, h_code_tok, _ = self.token_coattn(h_test_token, h_code_token)
        h_test_tok = h_test_tok.mean(dim=1)  # Pool to [batch, hidden]
        h_code_tok = h_code_tok.mean(dim=1)

        # Sentence-level
        h_test_sent, h_code_sent, _ = self.sentence_coattn(h_test_sent, h_code_sent)

        # Fuse
        h_test_final = self.fusion(torch.cat([h_test_tok, h_test_sent], dim=-1))
        h_code_final = self.fusion(torch.cat([h_code_tok, h_code_sent], dim=-1))

        return h_test_final, h_code_final
