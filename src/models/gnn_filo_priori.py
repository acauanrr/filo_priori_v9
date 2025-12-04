"""
GNN Filo-Priori Model - Adapted from Dual-Stream Model V8

This model adapts the Filo-Priori architecture that achieved APFD 0.7595
for Graph Neural Network benchmark datasets (Cora, CiteSeer, PubMed).

Key Adaptations:
- Node Feature Stream: Processes node features [N, F] instead of SBERT embeddings
- Structural Stream: GAT on the native graph structure (no need to build graph)
- Uncertainty Features: Extracted from GNN predictions instead of test history
- Multi-class: Supports multiple classes (not just binary Pass/Fail)

Architecture:
    Node Features [N, F] ──► NodeFeatureStream ──► [N, hidden_dim]
                                                          │
    Graph (edge_index) ──► StructuralStream (GAT) ──► [N, hidden_dim]
                                                          │
                                                          ▼
                                              CrossAttentionFusion
                                                          │
                                                          ▼
                                                    Classifier
                                                          │
                                                          ▼
                                                  Logits [N, C]

Author: Filo-Priori Team (Adapted for GNN Benchmarks)
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

try:
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    logging.warning("torch_geometric not available.")

logger = logging.getLogger(__name__)


class NodeFeatureStream(nn.Module):
    """
    Node Feature Stream: Processes node features.

    Replaces the SemanticStream (which processed SBERT embeddings)
    with a simple FFN that processes node features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of FFN layers with residual connections
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features [batch_size, input_dim]

        Returns:
            Processed features [batch_size, hidden_dim]
        """
        x = self.input_proj(x)

        for layer in self.layers:
            x = x + layer(x)  # Residual connection

        x = self.output_norm(x)
        return x


class StructuralStreamGAT(nn.Module):
    """
    GAT-based structural stream for GNN datasets.

    Uses the native graph structure from GNN benchmark datasets.
    No need to build co-failure graph - uses the citation/social network directly.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = 'elu'
    ):
        super().__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric is required")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()

        # First layer: hidden_dim -> hidden_dim * num_heads (concatenate)
        self.gat_layers.append(
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=True
            )
        )

        # Middle layers: hidden_dim * num_heads -> hidden_dim * num_heads
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim * num_heads,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )

        # Final layer: hidden_dim * num_heads -> hidden_dim (average)
        if num_layers > 1:
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim * num_heads,
                    out_channels=hidden_dim,
                    heads=1,
                    dropout=dropout,
                    concat=False
                )
            )

        # Activation
        if activation == 'elu':
            self.activation = F.elu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = F.gelu

        self.output_norm = nn.LayerNorm(hidden_dim)

        logger.info(f"Initialized StructuralStreamGAT:")
        logger.info(f"  - Input: [N, {input_dim}]")
        logger.info(f"  - {num_layers} GAT layers with {num_heads} heads")
        logger.info(f"  - Output: [N, {hidden_dim}]")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with graph attention.

        Args:
            x: Node features [N, input_dim]
            edge_index: Graph connectivity [2, E]
            edge_weights: Optional edge weights [E] (not used by GAT)

        Returns:
            Processed structural features [N, hidden_dim]
        """
        # Input projection
        x = self.input_proj(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.output_norm(x)
        return x


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion.

    Fuses node features with structural (GAT) features using attention.
    Same as original Filo-Priori V8.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # Cross-attention: node -> structural
        self.cross_attn_node2struct = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention: structural -> node
        self.cross_attn_struct2node = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm_node = nn.LayerNorm(hidden_dim)
        self.norm_struct = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(
        self,
        node_features: torch.Tensor,
        structural_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: fuse node and structural features

        Args:
            node_features: [batch_size, hidden_dim]
            structural_features: [batch_size, hidden_dim]

        Returns:
            fused_features: [batch_size, hidden_dim * 2]
        """
        # Add sequence dimension for attention
        node_seq = node_features.unsqueeze(1)      # [batch, 1, hidden_dim]
        struct_seq = structural_features.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Cross-attention: node attends to structural
        node_attended, _ = self.cross_attn_node2struct(
            query=node_seq,
            key=struct_seq,
            value=struct_seq
        )
        node_attended = node_attended.squeeze(1)
        node_enhanced = self.norm_node(node_features + node_attended)

        # Cross-attention: structural attends to node
        struct_attended, _ = self.cross_attn_struct2node(
            query=struct_seq,
            key=node_seq,
            value=node_seq
        )
        struct_attended = struct_attended.squeeze(1)
        structural_enhanced = self.norm_struct(structural_features + struct_attended)

        # Concatenate enhanced features
        fused = torch.cat([node_enhanced, structural_enhanced], dim=-1)
        fused = self.output_proj(fused)

        return fused


class GatedFusion(nn.Module):
    """
    Gated Fusion Unit for dynamic modality arbitration.

    Dynamically decides how much each stream contributes.
    Useful when structural features are sparse or noisy.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        structural_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: gated fusion

        Args:
            node_features: [batch_size, hidden_dim]
            structural_features: [batch_size, hidden_dim]

        Returns:
            fused_features: [batch_size, hidden_dim * 2]
        """
        # Compute gate
        gate_input = torch.cat([node_features, structural_features], dim=-1)
        z = self.gate(gate_input)

        # Gated fusion
        fused = z * node_features + (1 - z) * structural_features

        # Project output
        output = self.output_proj(fused)

        return output


class Classifier(nn.Module):
    """
    MLP classifier for multi-class classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64],
        num_classes: int = 7,
        dropout: float = 0.4
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GNNFiloPriori(nn.Module):
    """
    GNN Filo-Priori Model

    Adapted from the Dual-Stream Model V8 that achieved APFD 0.7595.

    Key differences from original:
    - Uses node features instead of SBERT embeddings
    - Uses native graph structure instead of co-failure graph
    - Supports multi-class classification

    Architecture:
        Node Features ──► NodeFeatureStream ──► [N, hidden_dim]
                                                       │
        Graph ──► StructuralStreamGAT ──► [N, hidden_dim]
                                                       │
                                                       ▼
                                           CrossAttentionFusion
                                                       │
                                                       ▼
                                                 Classifier
                                                       │
                                                       ▼
                                               Logits [N, C]
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_gat_heads: int = 4,
        num_gat_layers: int = 2,
        num_ffn_layers: int = 2,
        fusion_type: str = 'cross_attention',  # 'cross_attention' or 'gated'
        dropout: float = 0.3,
        classifier_hidden_dims: list = [128, 64]
    ):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Node Feature Stream (replaces SemanticStream)
        self.node_stream = NodeFeatureStream(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_ffn_layers,
            dropout=dropout
        )

        # Structural Stream (GAT)
        self.structural_stream = StructuralStreamGAT(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            num_heads=num_gat_heads,
            num_layers=num_gat_layers,
            dropout=dropout
        )

        # Fusion
        if fusion_type == 'gated':
            self.fusion = GatedFusion(
                hidden_dim=hidden_dim,
                dropout=dropout * 0.5
            )
        else:
            self.fusion = CrossAttentionFusion(
                hidden_dim=hidden_dim,
                num_heads=4,
                dropout=dropout * 0.5
            )

        # Classifier
        self.classifier = Classifier(
            input_dim=hidden_dim * 2,
            hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            dropout=dropout
        )

        logger.info("=" * 70)
        logger.info("GNN FILO-PRIORI MODEL INITIALIZED")
        logger.info("=" * 70)
        logger.info(f"Node Features: [N, {num_features}]")
        logger.info(f"Hidden Dim: {hidden_dim}")
        logger.info(f"GAT: {num_gat_layers} layers, {num_gat_heads} heads")
        logger.info(f"Fusion: {fusion_type}")
        logger.info(f"Classes: {num_classes}")
        logger.info("=" * 70)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features [N, num_features]
            edge_index: Graph connectivity [2, E]
            batch_mask: Optional mask for batch processing [batch_size]

        Returns:
            logits: [N, num_classes] or [batch_size, num_classes] if batch_mask provided
        """
        # Process both streams on full graph
        node_features = self.node_stream(x)
        structural_features = self.structural_stream(x, edge_index)

        # If batch_mask provided, select only those nodes
        if batch_mask is not None:
            node_features = node_features[batch_mask]
            structural_features = structural_features[batch_mask]

        # Fuse features
        fused = self.fusion(node_features, structural_features)

        # Classify
        logits = self.classifier(fused)

        return logits

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get intermediate embeddings for analysis or orphan scoring.

        Returns:
            node_features: [N, hidden_dim]
            structural_features: [N, hidden_dim]
            fused_features: [N, hidden_dim * 2]
        """
        node_features = self.node_stream(x)
        structural_features = self.structural_stream(x, edge_index)
        fused_features = self.fusion(node_features, structural_features)

        return node_features, structural_features, fused_features


def create_gnn_filo_priori(
    num_features: int,
    num_classes: int,
    config: Optional[Dict] = None
) -> GNNFiloPriori:
    """
    Factory function to create GNN Filo-Priori model.

    Args:
        num_features: Number of node features
        num_classes: Number of classes
        config: Optional configuration dict

    Returns:
        GNNFiloPriori instance
    """
    config = config or {}

    return GNNFiloPriori(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=config.get('hidden_dim', 256),
        num_gat_heads=config.get('num_gat_heads', 4),
        num_gat_layers=config.get('num_gat_layers', 2),
        num_ffn_layers=config.get('num_ffn_layers', 2),
        fusion_type=config.get('fusion_type', 'cross_attention'),
        dropout=config.get('dropout', 0.3),
        classifier_hidden_dims=config.get('classifier_hidden_dims', [128, 64])
    )


__all__ = [
    'NodeFeatureStream',
    'StructuralStreamGAT',
    'CrossAttentionFusion',
    'GatedFusion',
    'Classifier',
    'GNNFiloPriori',
    'create_gnn_filo_priori'
]
