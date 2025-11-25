"""
Graph Rewiring using Learned Link Prediction

Reconstructs the k-NN graph using learned edge scores instead of
semantic similarity alone. This creates a task-specific graph structure
that captures structural dependencies beyond pure semantic similarity.

Key Insight: Semantic similarity ≠ Structural relevance
Two test cases may be semantically distant but structurally coupled
(e.g., one change often causes both to fail).
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple
import logging
from pathlib import Path

from .link_prediction import LinkPredictor

logger = logging.getLogger(__name__)


def rewire_graph_with_link_predictor(
    x: torch.Tensor,
    original_edge_index: torch.Tensor,
    link_predictor: LinkPredictor,
    k: int,
    device: torch.device,
    batch_size: int = 1000,
    keep_original_ratio: float = 0.0,
    verbose: bool = True,
    candidate_multiplier: int = 20,
    scoring_device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rewire k-NN graph using learned link prediction scores.

    Args:
        x: Node features [num_nodes, input_dim]
        original_edge_index: Original k-NN edges [2, num_edges]
        link_predictor: Trained link prediction model
        k: Number of neighbors per node in rewired graph
        device: Device to run on
        batch_size: Batch size for scoring (to avoid OOM)
        keep_original_ratio: Ratio of original edges to keep (0.0 = all new, 1.0 = all original)
        verbose: Whether to print progress

    Returns:
        Tuple of (rewired_edge_index, edge_scores)
    """
    link_predictor.eval()
    num_nodes = x.size(0)

    if verbose:
        logger.info(f"Rewiring graph: {num_nodes} nodes, k={k}")

    # Move tensors to the encoding device and compute embeddings
    x = x.to(device)
    original_edge_index = original_edge_index.to(device)

    # Decide scoring device (default to CPU to avoid NVML/cuda allocator issues)
    if scoring_device is None:
        scoring_device = torch.device('cpu')

    with torch.no_grad():
        z = link_predictor.encode(x, original_edge_index)

    # For scoring and neighbor search we operate on CPU in a streaming manner
    z_cpu = z.detach().to(scoring_device)
    decoder_type = getattr(getattr(link_predictor, 'decoder', None), 'decoder_type', 'mlp')

    if verbose:
        logger.info(
            f"Scoring neighbors with decoder='{decoder_type}' on {scoring_device} "
            f"using candidate_multiplier={candidate_multiplier} and batch_size={batch_size}"
        )

    # Determine number of candidates per node for refinement
    L = max(k, min(num_nodes - 1, k * candidate_multiplier))

    # Build neighbors-of-neighbors candidate list from original graph (CPU)
    orig_cpu = original_edge_index.detach().to('cpu')
    adjacency = [[] for _ in range(num_nodes)]
    src_list = orig_cpu[0].tolist()
    dst_list = orig_cpu[1].tolist()
    for s, t in zip(src_list, dst_list):
        adjacency[s].append(t)

    # Prepare a dense candidate matrix [num_nodes, L] on scoring_device
    cand_idx = torch.empty(num_nodes, L, dtype=torch.long, device=scoring_device)

    rng = np.random.default_rng()
    for node in range(num_nodes):
        # union of neighbors-of-neighbors
        nn_set = set()
        for n1 in adjacency[node]:
            nn_set.update(adjacency[n1])
        nn_set.discard(node)

        candidates = list(nn_set)
        if len(candidates) >= L:
            chosen = rng.choice(candidates, size=L, replace=False)
        else:
            # fill remaining with random nodes (excluding self and duplicates)
            chosen = candidates
            need = L - len(chosen)
            if need > 0:
                pool_exclude = nn_set.union({node})
                # Oversample then filter to avoid long loops
                extra = []
                while len(extra) < need:
                    sample = rng.integers(0, num_nodes, size=need * 2)
                    for v in sample:
                        if v not in pool_exclude:
                            pool_exclude.add(int(v))
                            extra.append(int(v))
                            if len(extra) >= need:
                                break
                chosen = chosen + extra[:need]
            chosen = np.array(chosen, dtype=np.int64)

        cand_idx[node] = torch.from_numpy(chosen).to(scoring_device)

    if decoder_type in ('dot', 'distmult'):
        # Score selected candidates only
        rel_w = None
        if decoder_type == 'distmult':
            rel_w = link_predictor.decoder.relation_weights.to(scoring_device)

        new_edge_src = []
        new_edge_dst = []
        new_edge_scr = []

        for start in range(0, num_nodes, batch_size):
            end = min(start + batch_size, num_nodes)
            rows = torch.arange(start, end, device=scoring_device)
            z_i = z_cpu[start:end].unsqueeze(1).expand(-1, L, -1)  # [b, L, d]
            z_j = z_cpu[cand_idx[start:end]]  # [b, L, d]

            if rel_w is not None:
                z_i_eff = z_i * rel_w
                scores = (z_i_eff * z_j).sum(dim=-1)
            else:
                scores = (z_i * z_j).sum(dim=-1)

            # Top-k within candidates
            top_scores, top_idx_local = torch.topk(scores, k=k, dim=1, largest=True)
            top_cols = torch.gather(cand_idx[start:end], 1, top_idx_local)

            new_edge_src.append(rows.unsqueeze(1).expand(-1, k).reshape(-1))
            new_edge_dst.append(top_cols.reshape(-1))
            new_edge_scr.append(top_scores.reshape(-1))

        new_edge_index = torch.stack([
            torch.cat(new_edge_src, dim=0),
            torch.cat(new_edge_dst, dim=0)
        ], dim=0)
        new_edge_scores = torch.cat(new_edge_scr, dim=0)
    else:
        # Two-stage: shortlist by neighbors-of-neighbors, refine with MLP on candidates
        
        new_edge_src = []
        new_edge_dst = []
        new_edge_scr = []

        # Score candidates in manageable mini-batches
        with torch.no_grad():
            # We'll move the decoder to scoring_device for efficient CPU scoring
            # (keeps encoder on the original device)
            decoder = link_predictor.decoder.to(scoring_device)

            # Process i in batches to keep memory bounded
            for start in range(0, num_nodes, batch_size):
                end = min(start + batch_size, num_nodes)

                # Build edge_index for all candidates of this block
                batch_rows = torch.arange(start, end, device=scoring_device)
                batch_cands = cand_idx[start:end]  # [b, L]

                # Flatten edges
                rows_flat = batch_rows.unsqueeze(1).expand(-1, L).reshape(-1)
                cols_flat = batch_cands.reshape(-1)
                edge_index_block = torch.stack([rows_flat, cols_flat], dim=0)

                # Decode on scoring device using z_cpu
                block_scores = decoder(z_cpu, edge_index_block)
                block_scores = torch.sigmoid(block_scores)
                block_scores = block_scores.reshape(end - start, L)

                # Select top-k per i within candidates
                top_scores, top_idx_local = torch.topk(block_scores, k=k, dim=1, largest=True)
                top_cols = torch.gather(batch_cands, 1, top_idx_local)

                # Accumulate
                new_edge_src.append(batch_rows.unsqueeze(1).expand(-1, k).reshape(-1))
                new_edge_dst.append(top_cols.reshape(-1))
                new_edge_scr.append(top_scores.reshape(-1))

            # Stack results
            new_edge_index = torch.stack([
                torch.cat(new_edge_src, dim=0),
                torch.cat(new_edge_dst, dim=0)
            ], dim=0)
            new_edge_scores = torch.cat(new_edge_scr, dim=0)

        # Return decoder to original device (best effort)
        link_predictor.decoder.to(device)

    # Ensure tensors on the requested device for downstream
    if new_edge_index.device != device:
        new_edge_index = new_edge_index.to(device)
    if new_edge_scores.device != device:
        new_edge_scores = new_edge_scores.to(device)

    # Optional: Mix with original edges
    if keep_original_ratio > 0:
        if verbose:
            logger.info(f"Mixing with {keep_original_ratio*100:.1f}% original edges...")

        # Sample from original edges
        num_original_keep = int(original_edge_index.size(1) * keep_original_ratio)
        perm = torch.randperm(original_edge_index.size(1))[:num_original_keep]
        original_keep = original_edge_index[:, perm]

        # Get scores for original edges
        with torch.no_grad():
            original_scores = link_predictor.decode(z, original_keep)
            original_scores = torch.sigmoid(original_scores)

        # Combine
        new_edge_index = torch.cat([new_edge_index, original_keep], dim=1)
        new_edge_scores = torch.cat([new_edge_scores, original_scores])

    # Remove duplicates
    edge_set = set()
    unique_edges = []
    unique_scores = []

    for i in range(new_edge_index.size(1)):
        edge = (new_edge_index[0, i].item(), new_edge_index[1, i].item())
        if edge not in edge_set:
            edge_set.add(edge)
            unique_edges.append(edge)
            unique_scores.append(new_edge_scores[i].item())

    final_edge_index = torch.tensor(unique_edges, dtype=torch.long, device=device).t()
    final_edge_scores = torch.tensor(unique_scores, dtype=torch.float, device=device)

    if verbose:
        logger.info(f"Rewired graph: {final_edge_index.size(1)} edges (avg {final_edge_index.size(1)/num_nodes:.1f} per node)")
        logger.info(f"Edge score stats: mean={final_edge_scores.mean():.4f}, std={final_edge_scores.std():.4f}")

    return final_edge_index, final_edge_scores


def compute_graph_statistics(
    edge_index: torch.Tensor,
    num_nodes: int,
    name: str = "Graph"
) -> Dict[str, float]:
    """
    Compute statistics about the graph structure.

    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
        name: Name for logging

    Returns:
        Dictionary with graph statistics
    """
    num_edges = edge_index.size(1)

    # Compute degrees (ensure same device as edge_index)
    device = edge_index.device
    degrees = torch.zeros(num_nodes, dtype=torch.long, device=device)
    unique_sources = edge_index[0]
    degrees.index_add_(0, unique_sources, torch.ones_like(unique_sources))

    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': degrees.float().mean().item(),
        'min_degree': degrees.min().item(),
        'max_degree': degrees.max().item(),
        'std_degree': degrees.float().std().item()
    }

    logger.info(f"\n{name} Statistics:")
    logger.info(f"  Nodes: {stats['num_nodes']}")
    logger.info(f"  Edges: {stats['num_edges']}")
    logger.info(f"  Avg degree: {stats['avg_degree']:.2f}")
    logger.info(f"  Degree range: [{stats['min_degree']}, {stats['max_degree']}]")

    return stats


def save_rewired_graph(
    edge_index: torch.Tensor,
    edge_scores: torch.Tensor,
    output_path: str,
    metadata: Optional[Dict] = None
):
    """
    Save rewired graph to disk.

    Args:
        edge_index: Edge indices [2, num_edges]
        edge_scores: Edge scores [num_edges]
        output_path: Path to save file
        metadata: Optional metadata to include
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'edge_index': edge_index.cpu(),
        'edge_scores': edge_scores.cpu(),
        'num_edges': edge_index.size(1),
        'num_nodes': edge_index.max().item() + 1
    }

    if metadata is not None:
        save_dict['metadata'] = metadata

    torch.save(save_dict, output_path)
    logger.info(f"Saved rewired graph to {output_path}")


def load_rewired_graph(
    input_path: str,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Load rewired graph from disk.

    Args:
        input_path: Path to saved graph
        device: Device to load to

    Returns:
        Tuple of (edge_index, edge_scores, metadata)
    """
    data = torch.load(input_path, map_location=device)

    edge_index = data['edge_index'].to(device)
    edge_scores = data['edge_scores'].to(device)
    metadata = data.get('metadata', {})

    logger.info(f"Loaded rewired graph from {input_path}")
    logger.info(f"  Edges: {data['num_edges']}, Nodes: {data['num_nodes']}")

    return edge_index, edge_scores, metadata


def compare_graphs(
    original_edge_index: torch.Tensor,
    rewired_edge_index: torch.Tensor,
    num_nodes: int
) -> Dict[str, float]:
    """
    Compare original and rewired graphs.

    Args:
        original_edge_index: Original edges [2, num_edges_orig]
        rewired_edge_index: Rewired edges [2, num_edges_new]
        num_nodes: Number of nodes

    Returns:
        Dictionary with comparison metrics
    """
    # Convert to sets for comparison
    original_set = set()
    for i in range(original_edge_index.size(1)):
        edge = (original_edge_index[0, i].item(), original_edge_index[1, i].item())
        original_set.add(edge)

    rewired_set = set()
    for i in range(rewired_edge_index.size(1)):
        edge = (rewired_edge_index[0, i].item(), rewired_edge_index[1, i].item())
        rewired_set.add(edge)

    # Compute overlaps
    intersection = original_set & rewired_set
    union = original_set | rewired_set

    jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
    overlap_ratio = len(intersection) / len(original_set) if len(original_set) > 0 else 0.0

    num_added = len(rewired_set - original_set)
    num_removed = len(original_set - rewired_set)

    comparison = {
        'jaccard_similarity': jaccard,
        'overlap_ratio': overlap_ratio,
        'num_edges_added': num_added,
        'num_edges_removed': num_removed,
        'num_edges_kept': len(intersection)
    }

    logger.info("\nGraph Comparison:")
    logger.info(f"  Jaccard similarity: {jaccard:.4f}")
    logger.info(f"  Overlap ratio: {overlap_ratio:.4f}")
    logger.info(f"  Edges added: {num_added}")
    logger.info(f"  Edges removed: {num_removed}")
    logger.info(f"  Edges kept: {len(intersection)}")

    return comparison


def analyze_rewiring_quality(
    x: torch.Tensor,
    original_edge_index: torch.Tensor,
    rewired_edge_index: torch.Tensor,
    original_edge_weights: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Analyze the quality of graph rewiring.

    Compares semantic similarity of original vs rewired edges.

    Args:
        x: Node features (embeddings) [num_nodes, dim]
        original_edge_index: Original edges
        rewired_edge_index: Rewired edges
        original_edge_weights: Original edge weights (cosine similarity)

    Returns:
        Dictionary with quality metrics
    """
    # Compute cosine similarity for rewired edges
    row, col = rewired_edge_index[0], rewired_edge_index[1]
    x_normed = torch.nn.functional.normalize(x, p=2, dim=1)

    rewired_similarities = (x_normed[row] * x_normed[col]).sum(dim=1)

    # Compare with original
    if original_edge_weights is not None:
        quality = {
            'rewired_avg_similarity': rewired_similarities.mean().item(),
            'rewired_std_similarity': rewired_similarities.std().item(),
            'original_avg_similarity': original_edge_weights.mean().item(),
            'original_std_similarity': original_edge_weights.std().item()
        }

        logger.info("\nRewiring Quality:")
        logger.info(f"  Rewired avg similarity: {quality['rewired_avg_similarity']:.4f}")
        logger.info(f"  Original avg similarity: {quality['original_avg_similarity']:.4f}")

    else:
        quality = {
            'rewired_avg_similarity': rewired_similarities.mean().item(),
            'rewired_std_similarity': rewired_similarities.std().item()
        }

    return quality
