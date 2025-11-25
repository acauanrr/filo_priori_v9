"""
Standalone script for graph rewiring using link prediction.

This script performs the complete graph rewiring pipeline:
1. Load node features and original k-NN graph
2. Train link predictor (self-supervised)
3. Score all possible edges
4. Reconstruct graph with top-k predicted edges
5. Save rewired graph for use in main training

Usage:
    python run_graph_rewiring.py --config configs/rewiring_config.yaml
"""

import argparse
import yaml
import torch
import logging
from pathlib import Path

from src.phylogenetic.link_prediction import LinkPredictor
from src.phylogenetic.train_link_predictor import train_link_predictor_from_graph
from src.phylogenetic.graph_rewiring import (
    rewire_graph_with_link_predictor,
    compute_graph_statistics,
    save_rewired_graph,
    compare_graphs,
    analyze_rewiring_quality
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_graph_data(data_path: str, device: torch.device):
    """
    Load node features and edge index from saved graph.

    Args:
        data_path: Path to saved graph data
        device: Device to load to

    Returns:
        Tuple of (node_features, edge_index, edge_weights)
    """
    logger.info(f"Loading graph data from {data_path}")

    data = torch.load(data_path, map_location=device)

    x = data['node_features']  # [num_nodes, dim]
    edge_index = data['edge_index']  # [2, num_edges]
    edge_weights = data.get('edge_weights', None)  # [num_edges]

    logger.info(f"Loaded {x.size(0)} nodes with {x.size(1)}-dim features")
    logger.info(f"Loaded {edge_index.size(1)} edges")

    return x, edge_index, edge_weights


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")

    # Handle different config formats (backward compatibility)
    # Format 1: graph_data_path + output_dir (rewiring_015.yaml style)
    # Format 2: input_graph_path + output_graph_path (rewiring_017.yaml style)
    if 'graph_data_path' in config:
        graph_data_path = config['graph_data_path']
        output_dir = Path(config.get('output_dir', Path(graph_data_path).parent / 'rewiring'))
    elif 'input_graph_path' in config:
        graph_data_path = config['input_graph_path']
        if 'output_graph_path' in config:
            output_dir = Path(config['output_graph_path']).parent
        else:
            output_dir = Path(graph_data_path).parent / 'rewiring'
    else:
        raise ValueError("Config must contain either 'graph_data_path' or 'input_graph_path'")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load graph data
    x, original_edge_index, original_edge_weights = load_graph_data(
        graph_data_path,
        device
    )

    num_nodes = x.size(0)

    # Compute original graph statistics
    logger.info("\n" + "="*60)
    logger.info("ORIGINAL GRAPH STATISTICS")
    logger.info("="*60)
    original_stats = compute_graph_statistics(
        original_edge_index,
        num_nodes,
        name="Original k-NN Graph"
    )

    # Train link predictor
    logger.info("\n" + "="*60)
    logger.info("TRAINING LINK PREDICTOR")
    logger.info("="*60)

    link_predictor_config = config.get('link_predictor', {})

    link_predictor = train_link_predictor_from_graph(
        x=x,
        edge_index=original_edge_index,
        config=link_predictor_config,
        device=device,
        save_path=str(output_dir / 'link_predictor.pt')
    )

    # Rewire graph
    logger.info("\n" + "="*60)
    logger.info("REWIRING GRAPH")
    logger.info("="*60)

    rewiring_config = config.get('rewiring', {})

    rewired_edge_index, rewired_edge_scores = rewire_graph_with_link_predictor(
        x=x,
        original_edge_index=original_edge_index,
        link_predictor=link_predictor,
        k=rewiring_config.get('k', 10),
        device=device,
        batch_size=rewiring_config.get('batch_size', 1000),
        keep_original_ratio=rewiring_config.get('keep_original_ratio', 0.0),
        verbose=True,
        candidate_multiplier=rewiring_config.get('candidate_multiplier', 20),
        scoring_device=torch.device(rewiring_config['scoring_device']) if 'scoring_device' in rewiring_config else None
    )

    # Compute rewired graph statistics
    logger.info("\n" + "="*60)
    logger.info("REWIRED GRAPH STATISTICS")
    logger.info("="*60)
    rewired_stats = compute_graph_statistics(
        rewired_edge_index,
        num_nodes,
        name="Rewired Graph"
    )

    # Compare graphs
    logger.info("\n" + "="*60)
    logger.info("COMPARING GRAPHS")
    logger.info("="*60)
    comparison = compare_graphs(
        original_edge_index,
        rewired_edge_index,
        num_nodes
    )

    # Analyze rewiring quality
    logger.info("\n" + "="*60)
    logger.info("ANALYZING REWIRING QUALITY")
    logger.info("="*60)
    quality = analyze_rewiring_quality(
        x,
        original_edge_index,
        rewired_edge_index,
        original_edge_weights
    )

    # Save rewired graph
    logger.info("\n" + "="*60)
    logger.info("SAVING REWIRED GRAPH")
    logger.info("="*60)

    metadata = {
        'original_stats': original_stats,
        'rewired_stats': rewired_stats,
        'comparison': comparison,
        'quality': quality,
        'config': config
    }

    save_rewired_graph(
        rewired_edge_index,
        rewired_edge_scores,
        output_path=str(output_dir / 'rewired_graph.pt'),
        metadata=metadata
    )

    # Save summary report
    summary_path = output_dir / 'rewiring_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    logger.info(f"\nSaved summary report to {summary_path}")

    logger.info("\n" + "="*60)
    logger.info("GRAPH REWIRING COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info(f"\nOutput files:")
    logger.info(f"  - Rewired graph: {output_dir / 'rewired_graph.pt'}")
    logger.info(f"  - Link predictor: {output_dir / 'link_predictor.pt'}")
    logger.info(f"  - Summary: {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph rewiring with link prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to rewiring config file')

    args = parser.parse_args()
    main(args)
