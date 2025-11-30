"""
Filo-Priori V10 Hybrid Model Architecture.

This is the main model that integrates all V10 components:
1. CodeBERT + Co-Attention (semantic encoding)
2. Time-Decay Graph + GAT (structural encoding)
3. Heuristic Features (baseline signal)
4. Residual Fusion (combining everything)
5. LambdaRank optimization (direct APFD)

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────┐
    │                    FILO-PRIORI V10                          │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
    │  │ HEURISTIC   │  │  CODEBERT   │  │ TIME-DECAY  │         │
    │  │ FEATURES    │  │ + CO-ATTN   │  │ GRAPH + GAT │         │
    │  │    [6]      │  │   [768]     │  │    [256]    │         │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
    │         │                └──────────┬─────┘                │
    │         │                           │                      │
    │         │                    ┌──────┴──────┐               │
    │         │                    │NEURAL FUSION│               │
    │         │                    │   [256]     │               │
    │         │                    └──────┬──────┘               │
    │         │                           │                      │
    │         └───────────────────────────┼──────────────────────│
    │                                     │                      │
    │                              ┌──────┴──────┐               │
    │                              │  RESIDUAL   │               │
    │                              │   FUSION    │               │
    │                              │ α×h + (1-α)δ│               │
    │                              └──────┬──────┘               │
    │                                     │                      │
    │                              ┌──────┴──────┐               │
    │                              │   SCORE     │               │
    │                              │    [1]      │               │
    │                              └─────────────┘               │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

Key Design Decisions:
    1. Residual Learning: Neural network corrects heuristics, doesn't replace
    2. Hierarchical Fusion: Semantic + Structural first, then + Heuristics
    3. Time-Decay: Recent co-changes weighted more heavily
    4. LambdaRank: Optimizes ranking metric directly
"""

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from ..encoders.codebert_encoder import CodeBERTEncoder
from ..encoders.co_attention import CoAttention, MultiHeadCoAttention
from .residual_fusion import GatedResidualFusion, HierarchicalResidualFusion


@dataclass
class V10Config:
    """Configuration for Filo-Priori V10."""
    # Encoder
    encoder_model: str = "microsoft/codebert-base"
    encoder_dim: int = 768
    use_co_attention: bool = True
    co_attention_heads: int = 8
    co_attention_layers: int = 1

    # Graph
    graph_node_dim: int = 768  # Input dimension (from embeddings)
    graph_hidden_dim: int = 256
    graph_num_heads: int = 4
    graph_num_layers: int = 2
    graph_dropout: float = 0.1

    # Heuristics
    heuristic_dim: int = 6
    use_heuristics: bool = True
    initial_heuristic_weight: float = 0.7

    # Fusion
    fusion_type: str = "hierarchical"  # "simple", "gated", "hierarchical"
    fusion_hidden_dim: int = 256

    # Output
    output_dim: int = 1

    # Training
    dropout: float = 0.3


class FiloPrioriV10(nn.Module):
    """
    Filo-Priori V10 Hybrid Neuro-Symbolic Model.

    Combines:
    - CodeBERT for semantic understanding of tests/code
    - Time-Decay Graph for structural relationships
    - Heuristic features for strong baseline signal
    - Residual fusion for optimal combination

    Args:
        config: V10Config with model hyperparameters.

    Example:
        >>> config = V10Config()
        >>> model = FiloPrioriV10(config)
        >>> scores = model(
        ...     test_texts=["testLogin", "testPayment"],
        ...     code_texts=["UserService", "PaymentService"],
        ...     graph_data=graph,
        ...     heuristic_features=features
        ... )
    """

    def __init__(self, config: Optional[V10Config] = None):
        super().__init__()

        if config is None:
            config = V10Config()
        self.config = config

        # ======== Module 1: Semantic (CodeBERT) ========
        self.codebert = CodeBERTEncoder(
            model_name=config.encoder_model,
            output_dim=config.encoder_dim,
            pooling='cls'
        )

        # Co-Attention for test-code interaction
        if config.use_co_attention:
            if config.co_attention_layers > 1:
                self.co_attention = MultiHeadCoAttention(
                    hidden_dim=config.encoder_dim,
                    num_heads=config.co_attention_heads,
                    num_layers=config.co_attention_layers,
                    dropout=config.dropout
                )
            else:
                self.co_attention = CoAttention(
                    hidden_dim=config.encoder_dim,
                    num_heads=config.co_attention_heads,
                    dropout=config.dropout
                )
        else:
            self.co_attention = None

        # Semantic projection
        self.semantic_proj = nn.Sequential(
            nn.Linear(config.encoder_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        # ======== Module 2: Structural (GAT) ========
        self.gat_layers = nn.ModuleList()

        # First GAT layer
        self.gat_layers.append(
            GATConv(
                in_channels=config.graph_node_dim,
                out_channels=config.graph_hidden_dim,
                heads=config.graph_num_heads,
                concat=True,
                dropout=config.graph_dropout,
                edge_dim=1  # For edge weights
            )
        )

        # Additional GAT layers
        for _ in range(config.graph_num_layers - 1):
            self.gat_layers.append(
                GATConv(
                    in_channels=config.graph_hidden_dim * config.graph_num_heads,
                    out_channels=config.graph_hidden_dim,
                    heads=1,
                    concat=False,
                    dropout=config.graph_dropout,
                    edge_dim=1
                )
            )

        # Structural projection
        if config.graph_num_layers > 1:
            gat_output_dim = config.graph_hidden_dim
        else:
            gat_output_dim = config.graph_hidden_dim * config.graph_num_heads

        self.structural_proj = nn.Sequential(
            nn.Linear(gat_output_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        # ======== Module 3: Heuristics ========
        if config.use_heuristics:
            self.heuristic_encoder = nn.Sequential(
                nn.Linear(config.heuristic_dim, 32),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(32, 1)
            )
        else:
            self.heuristic_encoder = None

        # ======== Module 4: Fusion ========
        if config.fusion_type == "hierarchical":
            self.fusion = HierarchicalResidualFusion(
                heuristic_dim=config.heuristic_dim,
                semantic_dim=config.fusion_hidden_dim,
                structural_dim=config.fusion_hidden_dim,
                hidden_dim=config.fusion_hidden_dim,
                initial_heuristic_weight=config.initial_heuristic_weight
            )
        elif config.fusion_type == "gated":
            # First combine semantic + structural
            self.neural_combiner = nn.Sequential(
                nn.Linear(config.fusion_hidden_dim * 2, config.fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout)
            )
            self.fusion = GatedResidualFusion(
                heuristic_dim=config.heuristic_dim,
                neural_dim=config.fusion_hidden_dim,
                hidden_dim=64,
                initial_bias=1.0  # Trust heuristics initially
            )
        else:  # simple
            self.neural_combiner = nn.Sequential(
                nn.Linear(config.fusion_hidden_dim * 2, config.fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout)
            )
            self.scorer = nn.Linear(config.fusion_hidden_dim + config.heuristic_dim, 1)
            self.fusion = None

        # Trainable alpha for simple fusion
        if config.fusion_type == "simple":
            self.alpha = nn.Parameter(torch.tensor(config.initial_heuristic_weight))

    def forward(
        self,
        test_texts: Optional[List[str]] = None,
        code_texts: Optional[List[str]] = None,
        test_embeddings: Optional[torch.Tensor] = None,
        code_embeddings: Optional[torch.Tensor] = None,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
        graph_edge_weight: Optional[torch.Tensor] = None,
        heuristic_features: Optional[torch.Tensor] = None,
        node_indices: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through the hybrid model.

        Args:
            test_texts: List of test identifier strings (for CodeBERT).
            code_texts: List of code change strings (for CodeBERT).
            test_embeddings: Pre-computed test embeddings [batch, encoder_dim].
            code_embeddings: Pre-computed code embeddings [batch, encoder_dim].
            graph_x: Node features [num_nodes, node_dim].
            graph_edge_index: Edge connectivity [2, num_edges].
            graph_edge_weight: Edge weights [num_edges].
            heuristic_features: Heuristic features [batch, heuristic_dim].
            node_indices: Indices into graph for batch items [batch].
            return_intermediates: Whether to return intermediate representations.

        Returns:
            scores: Ranking scores [batch].
            intermediates: Optional dict with intermediate values.
        """
        batch_size = None
        intermediates = {} if return_intermediates else None

        # ======== Semantic Stream ========
        if test_embeddings is None and test_texts is not None:
            test_embeddings = self.codebert.encode(test_texts)
        if code_embeddings is None and code_texts is not None:
            code_embeddings = self.codebert.encode(code_texts)

        if test_embeddings is not None and code_embeddings is not None:
            batch_size = test_embeddings.shape[0]

            # Apply co-attention
            if self.co_attention is not None:
                if isinstance(self.co_attention, MultiHeadCoAttention):
                    h_test, h_code = self.co_attention(test_embeddings, code_embeddings)
                else:
                    h_test, h_code, attn_weights = self.co_attention(
                        test_embeddings, code_embeddings, return_weights=True
                    )
                    if intermediates is not None:
                        intermediates['attention_weights'] = attn_weights
            else:
                h_test = test_embeddings
                h_code = code_embeddings

            # Combine test and code
            h_semantic = self.semantic_proj(h_test + h_code)

            if intermediates is not None:
                intermediates['h_semantic'] = h_semantic
        else:
            h_semantic = None

        # ======== Structural Stream ========
        if graph_x is not None and graph_edge_index is not None:
            h_graph = graph_x

            for i, gat in enumerate(self.gat_layers):
                h_graph = gat(
                    h_graph,
                    graph_edge_index,
                    edge_attr=graph_edge_weight
                )
                if i < len(self.gat_layers) - 1:
                    h_graph = F.elu(h_graph)

            # Extract embeddings for batch items
            if node_indices is not None:
                h_structural = h_graph[node_indices]
            else:
                h_structural = h_graph

            h_structural = self.structural_proj(h_structural)

            if batch_size is None:
                batch_size = h_structural.shape[0]

            if intermediates is not None:
                intermediates['h_structural'] = h_structural
        else:
            h_structural = None

        # ======== Fusion ========
        if self.config.fusion_type == "hierarchical":
            # Use hierarchical fusion
            if h_semantic is None:
                h_semantic = torch.zeros(batch_size, self.config.fusion_hidden_dim,
                                        device=heuristic_features.device)
            if h_structural is None:
                h_structural = torch.zeros(batch_size, self.config.fusion_hidden_dim,
                                          device=heuristic_features.device)

            scores, fusion_info = self.fusion(
                h_heuristic=heuristic_features,
                h_semantic=h_semantic,
                h_structural=h_structural,
                return_intermediates=return_intermediates
            )

            if fusion_info is not None and intermediates is not None:
                intermediates.update(fusion_info)

        elif self.config.fusion_type == "gated":
            # Combine semantic + structural first
            if h_semantic is not None and h_structural is not None:
                h_neural = self.neural_combiner(
                    torch.cat([h_semantic, h_structural], dim=-1)
                )
            elif h_semantic is not None:
                h_neural = h_semantic
            elif h_structural is not None:
                h_neural = h_structural
            else:
                raise ValueError("Need at least semantic or structural features")

            scores, alpha = self.fusion(heuristic_features, h_neural, return_alpha=True)

            if intermediates is not None:
                intermediates['alpha'] = alpha

        else:  # simple
            # Simple concatenation + learned alpha
            if h_semantic is not None and h_structural is not None:
                h_neural = self.neural_combiner(
                    torch.cat([h_semantic, h_structural], dim=-1)
                )
            elif h_semantic is not None:
                h_neural = h_semantic
            elif h_structural is not None:
                h_neural = h_structural
            else:
                h_neural = torch.zeros(batch_size, self.config.fusion_hidden_dim,
                                       device=heuristic_features.device)

            # Concatenate all
            h_all = torch.cat([h_neural, heuristic_features], dim=-1)
            scores = self.scorer(h_all).squeeze(-1)

        return scores, intermediates

    def get_alpha(self) -> float:
        """Get current heuristic weight (alpha)."""
        if hasattr(self.fusion, 'get_alpha'):
            return self.fusion.get_alpha()
        elif hasattr(self, 'alpha'):
            return torch.sigmoid(self.alpha).item()
        else:
            return 0.5

    def freeze_encoder(self, num_layers: int = 6):
        """Freeze CodeBERT layers for fine-tuning efficiency."""
        self.codebert._freeze_encoder_layers(num_layers)

    def unfreeze_encoder(self):
        """Unfreeze all CodeBERT layers."""
        for param in self.codebert._encoder.parameters():
            param.requires_grad = True


class FiloPrioriV10Light(nn.Module):
    """
    Lightweight version of V10 without CodeBERT.

    Uses pre-computed embeddings instead of on-the-fly encoding.
    Faster but requires embeddings to be computed offline.
    """

    def __init__(self, config: Optional[V10Config] = None):
        super().__init__()

        if config is None:
            config = V10Config()
        self.config = config

        # Skip CodeBERT, use projections on pre-computed embeddings
        self.semantic_proj = nn.Sequential(
            nn.Linear(config.encoder_dim * 2, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim)
        )

        # GAT layers (same as full model)
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(
                in_channels=config.graph_node_dim,
                out_channels=config.graph_hidden_dim,
                heads=config.graph_num_heads,
                concat=True,
                dropout=config.graph_dropout,
                edge_dim=1
            )
        )

        self.structural_proj = nn.Sequential(
            nn.Linear(config.graph_hidden_dim * config.graph_num_heads, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        # Fusion
        self.fusion = HierarchicalResidualFusion(
            heuristic_dim=config.heuristic_dim,
            semantic_dim=config.fusion_hidden_dim,
            structural_dim=config.fusion_hidden_dim,
            hidden_dim=config.fusion_hidden_dim,
            initial_heuristic_weight=config.initial_heuristic_weight
        )

    def forward(
        self,
        semantic_embeddings: torch.Tensor,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_edge_weight: torch.Tensor,
        heuristic_features: torch.Tensor,
        node_indices: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with pre-computed embeddings."""
        # Semantic
        h_semantic = self.semantic_proj(semantic_embeddings)

        # Structural
        h_graph = self.gat_layers[0](graph_x, graph_edge_index, edge_attr=graph_edge_weight)
        h_graph = F.elu(h_graph)
        h_structural = self.structural_proj(h_graph[node_indices])

        # Fusion
        scores, _ = self.fusion(
            h_heuristic=heuristic_features,
            h_semantic=h_semantic,
            h_structural=h_structural
        )

        return scores
