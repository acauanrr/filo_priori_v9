"""
Dual-Stream Phylogenetic Transformer Model
Combines semantic stream and structural stream with cross-attention fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Union
from .cross_attention import CrossAttentionFusion
from .improved.classifier import ImprovedClassifier
from ..layers.gatv2 import GATv2Conv, ResidualGATv2Layer
from ..layers.denoising_gate import DenoisingGate, AdaptiveDenoisingGate, DenoisingGateWithNeighborDropout
from ..embeddings.field_fusion import create_field_fusion


class SemanticStream(nn.Module):
    """Semantic stream: processes node embeddings"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of transformer-style layers
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
            x: Input embeddings [batch_size, input_dim]

        Returns:
            Processed features [batch_size, hidden_dim]
        """
        x = self.input_proj(x)

        for layer in self.layers:
            x = x + layer(x)

        x = self.output_norm(x)
        return x


class StructuralStream(nn.Module):
    """
    Structural stream: processes graph structure using GNN-like operations
    Uses message passing on the k-NN graph

    Supports three layer types:
    - 'linear': Simple linear message passing (original implementation)
    - 'gat': Graph Attention Network (static attention)
    - 'gatv2': Graph Attention Network v2 (dynamic attention, recommended)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        aggregation: str = 'mean',
        use_edge_attention: bool = False,
        layer_type: str = 'linear',
        num_heads: int = 4,
        use_residual: bool = False,
        use_denoising_gate: bool = False,
        denoising_gate_type: str = 'mlp',
        denoising_gate_mode: str = 'basic',
        denoising_hard_threshold: Optional[float] = None,
        denoising_neighbor_dropout: float = 0.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.aggregation = aggregation
        self.use_edge_attention = use_edge_attention
        self.layer_type = layer_type
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.use_denoising_gate = use_denoising_gate

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Denoising gate (applied before GNN layers)
        if use_denoising_gate:
            if denoising_gate_mode == 'adaptive':
                self.denoising_gate = AdaptiveDenoisingGate(
                    hidden_dim=hidden_dim,
                    gate_type=denoising_gate_type,
                    dropout=dropout,
                    use_edge_features=True,
                    initial_threshold=0.1,
                    final_threshold=denoising_hard_threshold or 0.5,
                    warmup_epochs=10
                )
            elif denoising_gate_mode == 'neighbor_dropout':
                self.denoising_gate = DenoisingGateWithNeighborDropout(
                    hidden_dim=hidden_dim,
                    gate_type=denoising_gate_type,
                    dropout=dropout,
                    use_edge_features=True,
                    neighbor_dropout=denoising_neighbor_dropout,
                    hard_threshold=denoising_hard_threshold,
                    temperature=1.0
                )
            else:  # 'basic'
                self.denoising_gate = DenoisingGate(
                    hidden_dim=hidden_dim,
                    gate_type=denoising_gate_type,
                    dropout=dropout,
                    use_edge_features=True,
                    hard_threshold=denoising_hard_threshold,
                    temperature=1.0
                )
        else:
            self.denoising_gate = None

        # Build layers based on layer_type
        if layer_type == 'gatv2':
            # GATv2: Dynamic attention mechanism
            if use_residual:
                self.gnn_layers = nn.ModuleList([
                    ResidualGATv2Layer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        use_layer_norm=True
                    )
                    for _ in range(num_layers)
                ])
            else:
                self.gnn_layers = nn.ModuleList([
                    GATv2Conv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim // num_heads,
                        num_heads=num_heads,
                        concat=True,
                        dropout=dropout,
                        add_self_loops=True
                    )
                    for _ in range(num_layers)
                ])

            # Update layers for non-residual GATv2
            if not use_residual:
                self.update_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Dropout(dropout)
                    )
                    for _ in range(num_layers)
                ])
            else:
                self.update_layers = None

        elif layer_type == 'gat':
            # GAT: Static attention (placeholder - would need implementation)
            raise NotImplementedError("GAT layer type not yet implemented. Use 'gatv2' instead.")

        else:  # layer_type == 'linear' (default, backward compatible)
            # Message passing layers (original implementation)
            self.message_layers = nn.ModuleList([
                nn.Linear(hidden_dim * 2, hidden_dim)  # [source, target]
                for _ in range(num_layers)
            ])

            # Optional edge attention layers (scalar per edge)
            if self.use_edge_attention:
                self.attn_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.LeakyReLU(0.2),
                        nn.Linear(hidden_dim, 1)
                    ) for _ in range(num_layers)
                ])
            else:
                self.attn_layers = None

            self.update_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                )
                for _ in range(num_layers)
            ])

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with graph structure

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_weights: Optional edge weights [num_edges] (used for linear layer_type)

        Returns:
            Processed node features [num_nodes, hidden_dim]
        """
        x = self.input_proj(x)

        # Apply denoising gate if enabled
        if self.use_denoising_gate and self.denoising_gate is not None:
            gate_scores, filtered_edge_index = self.denoising_gate(x, edge_index, edge_weights)

            # Use filtered graph structure
            edge_index = filtered_edge_index

            # For soft gating, combine gate scores with edge weights
            if edge_weights is not None and len(gate_scores) == len(edge_weights):
                edge_weights = edge_weights * gate_scores
            else:
                edge_weights = gate_scores

        if self.layer_type == 'gatv2':
            # GATv2 layers: use graph attention
            # Note: edge_weights are not directly used in GATv2 as it learns
            # its own attention weights dynamically
            for layer_idx in range(self.num_layers):
                if self.use_residual:
                    # ResidualGATv2Layer handles residual connection internally
                    x = self.gnn_layers[layer_idx](x, edge_index)
                else:
                    # Manual residual connection
                    out = self.gnn_layers[layer_idx](x, edge_index)
                    out = self.update_layers[layer_idx](out)
                    x = x + out

        else:  # layer_type == 'linear'
            # Original linear message passing implementation
            for layer_idx in range(self.num_layers):
                # Message passing
                row, col = edge_index[0], edge_index[1]

                # Get source and target node features
                source_features = x[row]  # [num_edges, hidden_dim]
                target_features = x[col]  # [num_edges, hidden_dim]

                # Concatenate for message computation
                edge_feat = torch.cat([source_features, target_features], dim=-1)
                messages = self.message_layers[layer_idx](edge_feat)  # [num_edges, hidden_dim]

                # Edge attention (sigmoid gating per edge)
                if self.use_edge_attention and self.attn_layers is not None:
                    attn_logits = self.attn_layers[layer_idx](edge_feat).squeeze(-1)  # [num_edges]
                    attn_weights = torch.sigmoid(attn_logits)
                    messages = messages * attn_weights.unsqueeze(-1)

                # Apply edge weights if provided (similarity-based)
                if edge_weights is not None:
                    messages = messages * edge_weights.unsqueeze(-1)

                # Aggregate messages for each node
                num_nodes = x.size(0)
                aggregated = torch.zeros(num_nodes, self.hidden_dim, device=x.device)

                # Sum messages for each target node
                aggregated.index_add_(0, col, messages)

                # Normalize by number of neighbors
                degree = torch.zeros(num_nodes, device=x.device)
                degree.index_add_(0, col, torch.ones_like(col, dtype=torch.float))
                degree = torch.clamp(degree, min=1.0).unsqueeze(-1)

                if self.aggregation == 'mean' and not self.use_edge_attention:
                    aggregated = aggregated / degree

                # Update node features
                x = x + self.update_layers[layer_idx](aggregated)

        x = self.output_norm(x)
        return x

    def set_epoch(self, epoch: int):
        """
        Update epoch for adaptive denoising gate.
        Should be called at the beginning of each training epoch.
        """
        if self.use_denoising_gate and hasattr(self.denoising_gate, 'set_epoch'):
            self.denoising_gate.set_epoch(epoch)

    def get_denoising_stats(self) -> dict:
        """
        Get statistics about denoising gate performance.
        Useful for monitoring during training.
        """
        if not self.use_denoising_gate or self.denoising_gate is None:
            return {}

        stats = {}
        if hasattr(self.denoising_gate, 'get_current_threshold'):
            stats['threshold'] = self.denoising_gate.get_current_threshold()

        return stats


class DualStreamPhylogeneticTransformer(nn.Module):
    """
    Main model: Dual-Stream Phylogenetic Transformer
    Combines semantic and structural information with cross-attention
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.config = config
        model_config = config['model']
        embedding_config = config['embedding']

        # Get dimensions
        self.embedding_dim = embedding_config['embedding_dim']
        self.num_classes = model_config['classifier']['num_classes']

        # Multi-field embedding support
        self.use_multi_field = embedding_config.get('use_multi_field', False)
        if self.use_multi_field:
            # Create field fusion layer
            self.field_fusion = create_field_fusion(embedding_config)
            import logging
            logger = logging.getLogger(__name__)
            logger.info("✓ Multi-field embeddings enabled with field fusion layer")
        else:
            self.field_fusion = None

        # Semantic stream configuration
        semantic_config = model_config['semantic_stream']
        self.semantic_stream = SemanticStream(
            input_dim=self.embedding_dim,
            hidden_dim=semantic_config['hidden_dim'],
            num_layers=semantic_config['num_layers'],
            dropout=semantic_config['dropout'],
            activation=semantic_config.get('activation', 'gelu')
        )

        # Structural stream configuration
        structural_config = model_config['structural_stream']
        self.structural_stream = StructuralStream(
            input_dim=self.embedding_dim,
            hidden_dim=structural_config['hidden_dim'],
            num_layers=structural_config['num_gnn_layers'],
            dropout=structural_config['dropout'],
            aggregation=structural_config.get('aggregation', 'mean'),
            use_edge_attention=structural_config.get('use_edge_attention', False),
            layer_type=structural_config.get('layer_type', 'linear'),
            num_heads=structural_config.get('num_heads', 4),
            use_residual=structural_config.get('use_residual', False),
            use_denoising_gate=structural_config.get('use_denoising_gate', False),
            denoising_gate_type=structural_config.get('denoising_gate_type', 'mlp'),
            denoising_gate_mode=structural_config.get('denoising_gate_mode', 'basic'),
            denoising_hard_threshold=structural_config.get('denoising_hard_threshold', None),
            denoising_neighbor_dropout=structural_config.get('denoising_neighbor_dropout', 0.0)
        )

        # Cross-attention fusion
        cross_attn_config = model_config['cross_attention']
        self.cross_attention = CrossAttentionFusion(
            hidden_dim=cross_attn_config['hidden_dim'],
            num_heads=cross_attn_config['num_heads'],
            num_layers=cross_attn_config['num_layers'],
            dropout=cross_attn_config['dropout']
        )

        # Classification head
        classifier_config = model_config['classifier']
        fusion_dim = cross_attn_config['hidden_dim'] * 2  # Concatenated features

        # Build classifier based on type
        classifier_type = classifier_config.get('classifier_type', 'simple')

        if classifier_type == 'improved':
            # Use ImprovedClassifier with BatchNorm, Residual, and adaptive dropout
            dropout_base = classifier_config.get('dropout_base', 0.3)
            use_batch_norm = classifier_config.get('use_batch_norm', True)
            use_residual = classifier_config.get('use_residual', True)

            self.classifier = ImprovedClassifier(
                hidden_dim=fusion_dim,
                num_classes=self.num_classes,
                dropout=dropout_base,
                use_batch_norm=use_batch_norm,
                use_residual=use_residual
            )
        else:
            # Simple MLP classifier (backward compatible)
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, classifier_config['hidden_dim']),
                nn.GELU(),
                nn.Dropout(classifier_config['dropout']),
                nn.Linear(classifier_config['hidden_dim'], classifier_config['hidden_dim'] // 2),
                nn.GELU(),
                nn.Dropout(classifier_config['dropout']),
                nn.Linear(classifier_config['hidden_dim'] // 2, self.num_classes)
            )

    def forward(
        self,
        embeddings: Union[torch.Tensor, List[torch.Tensor]],
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
        batch_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            embeddings: Either single embeddings [batch_size, embedding_dim]
                       OR list of field embeddings [field1, field2, ...] each [batch_size, field_dim]
            edge_index: Graph edge indices [2, num_edges]
            edge_weights: Optional edge weights [num_edges]
            batch_indices: Batch assignment for each node (for batched graphs)

        Returns:
            Logits [batch_size, num_classes]
        """
        # Apply field fusion if multi-field embeddings provided
        if self.use_multi_field and isinstance(embeddings, list):
            # Fuse field embeddings into single embedding vector
            embeddings = self.field_fusion(embeddings)
        elif self.use_multi_field and not isinstance(embeddings, list):
            raise ValueError("Model configured for multi-field but received single embedding tensor")
        elif not self.use_multi_field and isinstance(embeddings, list):
            raise ValueError("Model not configured for multi-field but received list of embeddings")

        # Semantic stream: process node embeddings
        semantic_features = self.semantic_stream(embeddings)

        # Structural stream: process graph structure
        structural_features = self.structural_stream(
            embeddings,
            edge_index,
            edge_weights
        )

        # Cross-attention fusion
        fused_features = self.cross_attention(
            semantic_features,
            structural_features
        )

        # Classification
        logits = self.classifier(fused_features)

        return logits

    def get_embeddings(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get intermediate representations for analysis

        Returns:
            Dictionary with semantic, structural, and fused features
        """
        semantic_features = self.semantic_stream(embeddings)
        structural_features = self.structural_stream(embeddings, edge_index, edge_weights)
        fused_features = self.cross_attention(semantic_features, structural_features)

        return {
            'semantic': semantic_features,
            'structural': structural_features,
            'fused': fused_features
        }

    def set_epoch(self, epoch: int):
        """
        Update epoch for components that need it (e.g., adaptive denoising gate).
        Should be called at the beginning of each training epoch.
        """
        if hasattr(self.structural_stream, 'set_epoch'):
            self.structural_stream.set_epoch(epoch)

    def get_denoising_stats(self) -> dict:
        """
        Get statistics about denoising gate performance.
        Useful for monitoring during training.
        """
        if hasattr(self.structural_stream, 'get_denoising_stats'):
            return self.structural_stream.get_denoising_stats()
        return {}
