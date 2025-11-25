#!/usr/bin/env python3
"""
Phylogenetic Graph Visualization for Publication

Generates high-quality visualizations of the multi-edge phylogenetic graph
showing co-failure, co-success, and semantic relationships between test cases.

Author: Filo-Priori V8 Team
Date: 2025-11-14
"""

import pickle
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os

# Set publication-quality matplotlib settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

sns.set_palette("husl")


def load_graph(cache_path: str):
    """Load the multi-edge phylogenetic graph."""
    print(f"\n{'='*70}")
    print("LOADING PHYLOGENETIC GRAPH")
    print(f"{'='*70}")

    with open(cache_path, 'rb') as f:
        graph_data = pickle.load(f)

    edge_index = graph_data['edge_index']
    edge_types = graph_data['edge_types']
    edge_weights = graph_data['edge_weights']

    num_nodes = graph_data.get('num_nodes', edge_index.max() + 1)

    print(f"✓ Graph loaded successfully")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {edge_index.shape[1]}")

    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # Add edges by type
    edge_type_counter = Counter()

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        etype = edge_types[i]
        weight = edge_weights[i]

        edge_type_counter[etype] += 1

        # Add edge with attributes
        if G.has_edge(src, dst):
            # Multi-edge: add to existing
            G[src][dst]['types'].append(etype)
            G[src][dst]['weights'].append(weight)
        else:
            G.add_edge(src, dst, types=[etype], weights=[weight])

    print(f"\n  Edge Types:")
    print(f"    Co-failure:  {edge_type_counter[0]:6d} edges")
    print(f"    Co-success:  {edge_type_counter[1]:6d} edges")
    print(f"    Semantic:    {edge_type_counter[2]:6d} edges")

    return G, edge_type_counter


def visualize_graph_structure(G, output_dir: str):
    """Visualize overall graph structure."""
    print(f"\n{'='*70}")
    print("GRAPH STRUCTURE VISUALIZATION")
    print(f"{'='*70}")

    # Graph statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)

    # Degree distribution
    degrees = [G.degree(n) for n in G.nodes()]
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)

    print(f"\nGraph Statistics:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {num_edges}")
    print(f"  Density: {density:.4f}")
    print(f"  Avg Degree: {avg_degree:.2f}")
    print(f"  Max Degree: {max_degree}")

    # Check if connected
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        avg_path_length = nx.average_shortest_path_length(G)
        print(f"  Diameter: {diameter}")
        print(f"  Avg Path Length: {avg_path_length:.2f}")
    else:
        components = list(nx.connected_components(G))
        print(f"  Connected Components: {len(components)}")
        print(f"  Largest Component: {len(max(components, key=len))} nodes")

    # Clustering coefficient
    clustering = nx.average_clustering(G)
    print(f"  Avg Clustering: {clustering:.4f}")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Degree Distribution
    ax = axes[0, 0]
    ax.hist(degrees, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Node Degree')
    ax.set_ylabel('Frequency')
    ax.set_title('Degree Distribution')
    ax.axvline(avg_degree, color='red', linestyle='--', label=f'Mean: {avg_degree:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Degree Distribution (Log-Log)
    ax = axes[0, 1]
    degree_counts = Counter(degrees)
    degrees_sorted = sorted(degree_counts.keys())
    counts = [degree_counts[d] for d in degrees_sorted]
    ax.loglog(degrees_sorted, counts, 'o', markersize=5, alpha=0.6)
    ax.set_xlabel('Node Degree (log scale)')
    ax.set_ylabel('Frequency (log scale)')
    ax.set_title('Degree Distribution (Log-Log)')
    ax.grid(True, alpha=0.3, which='both')

    # 3. Clustering Coefficient Distribution
    ax = axes[1, 0]
    clustering_coeffs = list(nx.clustering(G).values())
    ax.hist(clustering_coeffs, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Clustering Coefficient')
    ax.set_ylabel('Frequency')
    ax.set_title('Clustering Coefficient Distribution')
    ax.axvline(clustering, color='red', linestyle='--', label=f'Mean: {clustering:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Component Size Distribution (if disconnected)
    ax = axes[1, 1]
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        comp_sizes = [len(c) for c in components]
        ax.hist(comp_sizes, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Component Size')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Component Size Distribution ({len(components)} components)')
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'Graph is fully connected',
                ha='center', va='center', fontsize=12)
        ax.set_title('Component Analysis')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'graph_structure_analysis.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\n✓ Graph structure visualization saved: {output_path}")
    plt.close()


def visualize_edge_types(edge_type_counter, output_dir: str):
    """Visualize distribution of edge types."""
    print(f"\n{'='*70}")
    print("EDGE TYPE DISTRIBUTION")
    print(f"{'='*70}")

    edge_names = ['Co-failure', 'Co-success', 'Semantic']
    edge_counts = [edge_type_counter[i] for i in range(3)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    ax = axes[0]
    bars = ax.bar(edge_names, edge_counts, color=['#e74c3c', '#27ae60', '#3498db'],
                   edgecolor='black', alpha=0.8)
    ax.set_ylabel('Number of Edges')
    ax.set_title('Edge Type Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10)

    # Pie chart
    ax = axes[1]
    colors = ['#e74c3c', '#27ae60', '#3498db']
    explode = (0.05, 0.05, 0.05)
    wedges, texts, autotexts = ax.pie(edge_counts, labels=edge_names, autopct='%1.1f%%',
                                        colors=colors, explode=explode, startangle=90)
    ax.set_title('Edge Type Proportion')

    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'edge_type_distribution.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Edge type distribution saved: {output_path}")
    plt.close()


def visualize_graph_sample(G, output_dir: str, sample_size: int = 100):
    """Visualize a sample of the graph with edge types."""
    print(f"\n{'='*70}")
    print(f"GRAPH SAMPLE VISUALIZATION ({sample_size} nodes)")
    print(f"{'='*70}")

    # Sample nodes with highest degree (most connected)
    degrees = dict(G.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:sample_size]
    sample_nodes = [n for n, d in top_nodes]

    # Create subgraph
    G_sample = G.subgraph(sample_nodes).copy()

    print(f"\nSample Statistics:")
    print(f"  Nodes: {G_sample.number_of_nodes()}")
    print(f"  Edges: {G_sample.number_of_edges()}")

    # Separate edges by dominant type
    cofailure_edges = []
    cosuccess_edges = []
    semantic_edges = []

    for u, v, data in G_sample.edges(data=True):
        types = data['types']
        # Dominant type (first one)
        if 0 in types:
            cofailure_edges.append((u, v))
        elif 1 in types:
            cosuccess_edges.append((u, v))
        else:
            semantic_edges.append((u, v))

    # Layout
    pos = nx.spring_layout(G_sample, k=2, iterations=50, seed=42)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))

    # Draw nodes
    nx.draw_networkx_nodes(G_sample, pos, node_size=100,
                           node_color='lightgray',
                           edgecolors='black', linewidths=0.5,
                           alpha=0.9, ax=ax)

    # Draw edges by type
    if cofailure_edges:
        nx.draw_networkx_edges(G_sample, pos, edgelist=cofailure_edges,
                               edge_color='#e74c3c', width=1.5, alpha=0.6,
                               label='Co-failure', ax=ax)

    if cosuccess_edges:
        nx.draw_networkx_edges(G_sample, pos, edgelist=cosuccess_edges,
                               edge_color='#27ae60', width=1.0, alpha=0.4,
                               label='Co-success', ax=ax)

    if semantic_edges:
        nx.draw_networkx_edges(G_sample, pos, edgelist=semantic_edges,
                               edge_color='#3498db', width=0.8, alpha=0.3,
                               label='Semantic', ax=ax, style='dashed')

    ax.set_title(f'Phylogenetic Graph Sample (Top {sample_size} Most Connected Nodes)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'graph_sample_visualization.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Graph sample visualization saved: {output_path}")
    plt.close()


def analyze_edge_weight_distribution(G, output_dir: str):
    """Analyze distribution of edge weights by type."""
    print(f"\n{'='*70}")
    print("EDGE WEIGHT ANALYSIS")
    print(f"{'='*70}")

    cofailure_weights = []
    cosuccess_weights = []
    semantic_weights = []

    for u, v, data in G.edges(data=True):
        types = data['types']
        weights = data['weights']

        for etype, weight in zip(types, weights):
            if etype == 0:
                cofailure_weights.append(weight)
            elif etype == 1:
                cosuccess_weights.append(weight)
            else:
                semantic_weights.append(weight)

    print(f"\nEdge Weight Statistics:")
    print(f"  Co-failure:  {len(cofailure_weights):6d} edges, "
          f"mean={np.mean(cofailure_weights):.3f}, "
          f"std={np.std(cofailure_weights):.3f}")
    print(f"  Co-success:  {len(cosuccess_weights):6d} edges, "
          f"mean={np.mean(cosuccess_weights):.3f}, "
          f"std={np.std(cosuccess_weights):.3f}")
    print(f"  Semantic:    {len(semantic_weights):6d} edges, "
          f"mean={np.mean(semantic_weights):.3f}, "
          f"std={np.std(semantic_weights):.3f}")

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Co-failure weights
    ax = axes[0]
    ax.hist(cofailure_weights, bins=50, color='#e74c3c', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Edge Weight')
    ax.set_ylabel('Frequency')
    ax.set_title('Co-failure Edge Weights')
    ax.axvline(np.mean(cofailure_weights), color='darkred', linestyle='--',
               label=f'Mean: {np.mean(cofailure_weights):.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Co-success weights
    ax = axes[1]
    ax.hist(cosuccess_weights, bins=50, color='#27ae60', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Edge Weight')
    ax.set_ylabel('Frequency')
    ax.set_title('Co-success Edge Weights')
    ax.axvline(np.mean(cosuccess_weights), color='darkgreen', linestyle='--',
               label=f'Mean: {np.mean(cosuccess_weights):.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Semantic weights
    ax = axes[2]
    ax.hist(semantic_weights, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Edge Weight')
    ax.set_ylabel('Frequency')
    ax.set_title('Semantic Edge Weights')
    ax.axvline(np.mean(semantic_weights), color='darkblue', linestyle='--',
               label=f'Mean: {np.mean(semantic_weights):.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'edge_weight_distribution.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Edge weight distribution saved: {output_path}")
    plt.close()


def main():
    """Main visualization pipeline."""
    cache_path = "cache/multi_edge_graph.pkl"
    output_dir = "results/publication/visualizations"

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("PHYLOGENETIC GRAPH VISUALIZATION FOR PUBLICATION")
    print("="*70)

    # Load graph
    G, edge_type_counter = load_graph(cache_path)

    # Generate visualizations
    visualize_graph_structure(G, output_dir)
    visualize_edge_types(edge_type_counter, output_dir)
    visualize_graph_sample(G, output_dir, sample_size=150)
    analyze_edge_weight_distribution(G, output_dir)

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. graph_structure_analysis.png")
    print("  2. edge_type_distribution.png")
    print("  3. graph_sample_visualization.png")
    print("  4. edge_weight_distribution.png")
    print("="*70)


if __name__ == "__main__":
    main()
