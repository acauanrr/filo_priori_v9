"""
Bidirectional Cross-Attention Fusion

Fusão melhorada que permite interação bidirecional entre fluxos
semântico e estrutural através de cross-attention.
"""

import torch
import torch.nn as nn


class BidirectionalCrossAttentionFusion(nn.Module):
    """
    Fusão por atenção cruzada bidirecional.

    Permite que semantic e structural streams atendam um ao outro
    de forma simétrica, criando uma representação fundida rica.

    Args:
        hidden_dim: Dimensão oculta dos fluxos
        num_heads: Número de attention heads
        dropout: Taxa de dropout
        use_feedforward: Se True, adiciona FFN após fusão
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_feedforward: bool = True
    ):
        super().__init__()

        # Cross-attention: semantic → structural
        self.cross_attn_sem2struct = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention: structural → semantic
        self.cross_attn_struct2sem = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Layer norms
        self.norm_sem = nn.LayerNorm(hidden_dim)
        self.norm_struct = nn.LayerNorm(hidden_dim)

        # Optional feed-forward networks
        self.use_feedforward = use_feedforward
        if use_feedforward:
            self.ffn_sem = self._build_ffn(hidden_dim, dropout)
            self.ffn_struct = self._build_ffn(hidden_dim, dropout)
            self.norm_ffn_sem = nn.LayerNorm(hidden_dim)
            self.norm_ffn_struct = nn.LayerNorm(hidden_dim)

        # Gating mechanism para fusão final
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh()
        )

    def _build_ffn(self, hidden_dim, dropout):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, semantic_features, structural_features):
        """
        Args:
            semantic_features: [batch, hidden_dim]
            structural_features: [batch, hidden_dim]

        Returns:
            fused_features: [batch, hidden_dim]
            attention_info: dict com attention weights
        """
        # Adicionar dimensão de sequência para MultiheadAttention
        sem_seq = semantic_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        struct_seq = structural_features.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Cross-attention: semantic attends to structural
        sem_attended, sem_attn_weights = self.cross_attn_sem2struct(
            query=sem_seq,
            key=struct_seq,
            value=struct_seq
        )
        sem_attended = sem_attended.squeeze(1)

        # Residual + norm
        semantic_enhanced = self.norm_sem(semantic_features + sem_attended)

        # Cross-attention: structural attends to semantic
        struct_attended, struct_attn_weights = self.cross_attn_struct2sem(
            query=struct_seq,
            key=sem_seq,
            value=sem_seq
        )
        struct_attended = struct_attended.squeeze(1)

        # Residual + norm
        structural_enhanced = self.norm_struct(structural_features + struct_attended)

        # Optional feed-forward
        if self.use_feedforward:
            semantic_enhanced = self.norm_ffn_sem(
                semantic_enhanced + self.ffn_sem(semantic_enhanced)
            )
            structural_enhanced = self.norm_ffn_struct(
                structural_enhanced + self.ffn_struct(structural_enhanced)
            )

        # Fusão final com gating
        concat = torch.cat([semantic_enhanced, structural_enhanced], dim=-1)
        fused = self.fusion_gate(concat)

        # Informações de atenção para interpretabilidade
        attention_info = {
            'sem_attn_weights': sem_attn_weights.squeeze(1),
            'struct_attn_weights': struct_attn_weights.squeeze(1),
            'semantic_enhanced': semantic_enhanced,
            'structural_enhanced': structural_enhanced
        }

        return fused, attention_info


class MultiLayerCrossAttentionFusion(nn.Module):
    """
    Múltiplas camadas de cross-attention bidirecional.

    Para fusão ainda mais profunda e interativa.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            BidirectionalCrossAttentionFusion(
                hidden_dim, num_heads, dropout, use_feedforward=True
            )
            for _ in range(num_layers)
        ])

    def forward(self, semantic_features, structural_features):
        """
        Args:
            semantic_features: [batch, hidden_dim]
            structural_features: [batch, hidden_dim]

        Returns:
            fused_features: [batch, hidden_dim]
            all_attention_info: list de dicts com attention weights de cada camada
        """
        all_attention_info = []

        for layer in self.layers:
            fused, attn_info = layer(semantic_features, structural_features)

            # Atualizar features para próxima camada
            semantic_features = fused
            structural_features = fused

            all_attention_info.append(attn_info)

        return fused, all_attention_info
