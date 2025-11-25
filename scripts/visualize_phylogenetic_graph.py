"""
Visualize Phylogenetic Graph for Filo-Priori V8

This script visualizes the phylogenetic graph structure, showing:
1. Batch subgraphs with labels and predictions
2. GAT attention weight heatmaps
3. Graph statistics and analysis

Usage:
    python scripts/visualize_phylogenetic_graph.py \
        --config configs/experiment.yaml \
        --batch_size 32 \
        --output visualizations/

Author: Filo-Priori V8 Team
Date: 2025-11-13
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import yaml
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

from preprocessing.data_loader import DataLoader
from phylogenetic.phylogenetic_graph_builder import build_phylogenetic_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_batch_subgraph(
    tc_keys: List[str],
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    labels: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    structural_features: Optional[np.ndarray] = None,
    save_path: str = "graph_visualization.png"
):
    """
    Visualize phylogenetic graph subgraph for a batch

    Args:
        tc_keys: List of TC_Key strings
        edge_index: [2, E] edge connectivity
        edge_weights: [E] edge weights
        labels: [N] ground truth labels (0=Fail, 1=Pass)
        predictions: [N] predicted labels (optional)
        structural_features: [N, 6] structural features (optional)
        save_path: Output path for visualization
    """
    # Create NetworkX graph
    G = nx.Graph()

    # Add nodes
    for i, tc_key in enumerate(tc_keys):
        label = "Fail" if labels[i] == 0 else "Pass"

        node_attrs = {
            'tc_key': tc_key,
            'label': label,
            'label_idx': int(labels[i])
        }

        if predictions is not None:
            pred = "Fail" if predictions[i] == 0 else "Pass"
            node_attrs['prediction'] = pred
            node_attrs['correct'] = (label == pred)

        if structural_features is not None:
            node_attrs['features'] = structural_features[i]

        G.add_node(i, **node_attrs)

    # Add edges
    edge_index_np = edge_index.cpu().numpy() if torch.is_tensor(edge_index) else edge_index
    edge_weights_np = edge_weights.cpu().numpy() if torch.is_tensor(edge_weights) else edge_weights

    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[:, i]
        weight = float(edge_weights_np[i])
        G.add_edge(int(src), int(dst), weight=weight)

    # Layout
    logger.info(f"Computing graph layout for {G.number_of_nodes()} nodes...")
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

    # Create figure
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # ==== PLOT 1: Graph with Labels ====
    ax1 = fig.add_subplot(gs[0, :2])

    # Node colors by label (red=Fail, green=Pass)
    node_colors = ['#FF6B6B' if labels[i] == 0 else '#51CF66' for i in range(len(tc_keys))]

    # Draw edges (thickness = weight)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    nx.draw_networkx_edges(G, pos,
                          width=[w*8 for w in weights],
                          alpha=0.2,
                          edge_color='gray',
                          ax=ax1)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=1200,
                          alpha=0.8,
                          edgecolors='black',
                          linewidths=2,
                          ax=ax1)

    # Node labels
    labels_dict = {i: f"{tc_keys[i][:10]}\n{G.nodes[i]['label']}"
                   for i in range(len(tc_keys))}
    nx.draw_networkx_labels(G, pos, labels_dict, font_size=9, font_weight='bold', ax=ax1)

    ax1.set_title(f"Phylogenetic Graph Subgraph (Batch Size: {len(tc_keys)})\n"
                  f"Red=Fail, Green=Pass | Edge thickness ∝ co-failure probability",
                  fontsize=16, fontweight='bold')
    ax1.axis('off')

    # ==== PLOT 2: Predictions (if available) ====
    if predictions is not None:
        ax2 = fig.add_subplot(gs[0, 2])

        # Node colors by correctness (green=correct, red=incorrect)
        pred_colors = []
        for i in range(len(tc_keys)):
            correct = G.nodes[i]['correct']
            pred_colors.append('#51CF66' if correct else '#FF6B6B')

        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.2, edge_color='gray', ax=ax2)
        nx.draw_networkx_nodes(G, pos,
                              node_color=pred_colors,
                              node_size=800,
                              alpha=0.8,
                              edgecolors='black',
                              linewidths=2,
                              ax=ax2)

        pred_labels = {i: "✓" if G.nodes[i]['correct'] else "✗"
                       for i in range(len(tc_keys))}
        nx.draw_networkx_labels(G, pos, pred_labels, font_size=12, font_weight='bold', ax=ax2)

        # Accuracy
        accuracy = sum(G.nodes[i]['correct'] for i in range(len(tc_keys))) / len(tc_keys)
        ax2.set_title(f"Prediction Correctness\n"
                      f"Green=Correct, Red=Incorrect\n"
                      f"Accuracy: {accuracy*100:.1f}%",
                      fontsize=12, fontweight='bold')
        ax2.axis('off')

    # ==== PLOT 3: Graph Statistics ====
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')

    stats_text = f"""GRAPH STATISTICS

Nodes: {G.number_of_nodes()}
Edges: {G.number_of_edges()}
Density: {nx.density(G):.4f}

Degree Statistics:
  - Avg degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}
  - Max degree: {max(dict(G.degree()).values())}
  - Min degree: {min(dict(G.degree()).values())}

Edge Weights:
  - Mean: {np.mean(weights):.4f}
  - Std: {np.std(weights):.4f}
  - Min: {np.min(weights):.4f}
  - Max: {np.max(weights):.4f}

Connected Components: {nx.number_connected_components(G)}
Isolated Nodes: {sum(1 for c in nx.connected_components(G) if len(c) == 1)}

Label Distribution:
  - Fail: {sum(1 for i in range(len(tc_keys)) if labels[i] == 0)} ({sum(1 for i in range(len(tc_keys)) if labels[i] == 0)/len(tc_keys)*100:.1f}%)
  - Pass: {sum(1 for i in range(len(tc_keys)) if labels[i] == 1)} ({sum(1 for i in range(len(tc_keys)) if labels[i] == 1)/len(tc_keys)*100:.1f}%)
"""

    ax3.text(0.05, 0.95, stats_text,
            transform=ax3.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # ==== PLOT 4: Degree Distribution ====
    ax4 = fig.add_subplot(gs[1, 1])

    degrees = [G.degree(i) for i in range(len(tc_keys))]
    ax4.hist(degrees, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Node Degree', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Degree Distribution', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)

    # ==== PLOT 5: Edge Weight Distribution ====
    ax5 = fig.add_subplot(gs[1, 2])

    ax5.hist(weights, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Edge Weight', fontsize=12)
    ax5.set_ylabel('Count', fontsize=12)
    ax5.set_title('Edge Weight Distribution', fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3)

    plt.suptitle(f'Phylogenetic Graph Analysis', fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved graph visualization to {save_path}")

    plt.close()

    return G


def visualize_attention_weights(
    edge_index: torch.Tensor,
    attention_weights: torch.Tensor,
    num_nodes: int,
    save_path: str = "attention_heatmap.png"
):
    """
    Visualize GAT attention weights as heatmap

    Args:
        edge_index: [2, E] edge connectivity
        attention_weights: [E, num_heads] attention weights from GAT
        num_nodes: Number of nodes
        save_path: Output path
    """
    num_heads = attention_weights.shape[1]

    fig, axes = plt.subplots(1, num_heads, figsize=(6*num_heads, 5))

    if num_heads == 1:
        axes = [axes]

    for head in range(num_heads):
        ax = axes[head]

        # Create attention matrix [N, N]
        attn_matrix = np.zeros((num_nodes, num_nodes))

        edge_index_np = edge_index.cpu().numpy()
        attn_np = attention_weights.cpu().numpy()

        for i in range(edge_index.shape[1]):
            src, dst = edge_index_np[:, i]
            attn_matrix[src, dst] = attn_np[i, head]

        # Heatmap
        im = ax.imshow(attn_matrix, cmap='hot', interpolation='nearest', aspect='auto')
        ax.set_title(f'GAT Attention Head {head+1}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Node', fontsize=12)
        ax.set_ylabel('Source Node', fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('GAT Multi-Head Attention Weights', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved attention heatmap to {save_path}")
    plt.close()


def analyze_graph_patterns(
    graph_builder,
    df: pd.DataFrame,
    num_samples: int = 5
):
    """
    Analyze interesting graph patterns

    Args:
        graph_builder: PhylogeneticGraphBuilder instance
        df: DataFrame with test data
        num_samples: Number of sample subgraphs to visualize
    """
    logger.info("\n" + "="*70)
    logger.info("ANALYZING GRAPH PATTERNS")
    logger.info("="*70)

    # Get global graph statistics
    stats = graph_builder.get_graph_statistics()
    logger.info(f"\nGlobal Graph Statistics:")
    logger.info(f"  Total nodes: {stats['num_nodes']}")
    logger.info(f"  Total edges: {stats['num_edges']}")
    logger.info(f"  Avg degree: {stats['avg_degree']:.2f}")
    logger.info(f"  Avg edge weight: {stats['avg_edge_weight']:.4f}")

    # Sample interesting patterns
    logger.info(f"\nSampling {num_samples} interesting subgraphs...")

    # Pattern 1: High co-failure cluster (tests that often fail together)
    # Pattern 2: Isolated nodes (new tests)
    # Pattern 3: Mixed (fail and pass)

    # For now, sample random batches
    batch_size = 32

    for i in range(num_samples):
        # Random sample
        sample_df = df.sample(n=min(batch_size, len(df)), random_state=42+i)
        tc_keys = sample_df['TC_Key'].tolist()
        labels = torch.tensor(sample_df['label'].values)

        # Get subgraph
        edge_index, edge_weights = graph_builder.get_edge_index_and_weights(
            tc_keys,
            return_torch=True
        )

        # Visualize
        output_path = f"visualizations/subgraph_sample_{i+1}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        G = visualize_batch_subgraph(
            tc_keys=tc_keys,
            edge_index=edge_index,
            edge_weights=edge_weights,
            labels=labels,
            save_path=output_path
        )

        logger.info(f"  Sample {i+1}: {G.number_of_nodes()} nodes, "
                   f"{G.number_of_edges()} edges, "
                   f"{nx.number_connected_components(G)} components")


def main():
    parser = argparse.ArgumentParser(description='Visualize Phylogenetic Graph')
    parser.add_argument('--config', type=str, default='configs/experiment.yaml',
                       help='Path to configuration file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for subgraph visualization')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of sample subgraphs to visualize')
    parser.add_argument('--output', type=str, default='visualizations/',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    data_loader = DataLoader(config)
    data_dict = data_loader.prepare_dataset()

    df_train = data_dict['train']

    # Build phylogenetic graph
    logger.info("\nBuilding phylogenetic graph...")
    graph_config = config.get('phylogenetic', {})

    graph_builder = build_phylogenetic_graph(
        df_train,
        graph_type=graph_config.get('graph_type', 'co_failure'),
        min_co_occurrences=graph_config.get('min_co_occurrences', 2),
        weight_threshold=graph_config.get('weight_threshold', 0.1),
        cache_path=graph_config.get('cache_path')
    )

    # Analyze patterns
    analyze_graph_patterns(
        graph_builder,
        df_train,
        num_samples=args.num_samples
    )

    logger.info("\n" + "="*70)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Output saved to: {args.output}")


if __name__ == '__main__':
    main()
