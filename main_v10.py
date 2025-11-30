#!/usr/bin/env python3
"""
Filo-Priori V10 Training Pipeline.

This script implements the complete training pipeline for V10,
including:
1. Data loading and preprocessing
2. Time-decay graph construction
3. CodeBERT encoding with co-attention
4. Heuristic feature extraction
5. Hybrid model training with LambdaRank
6. Evaluation against baselines

Usage:
    python main_v10.py --config configs/experiment_v10_rtptorrent.yaml

Author: Filo-Priori Team
Version: 10.0.0
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# V10 imports
from src.v10.models.hybrid_model import FiloPrioriV10, V10Config
from src.v10.graphs.time_decay_builder import TimeDecayGraphBuilder
from src.v10.graphs.co_change_miner import RTPTorrentCoChangeMiner
from src.v10.encoders.codebert_encoder import CodeBERTEncoder
from src.v10.encoders.tokenizer import JavaTestTokenizer
from src.v10.features.heuristic_features import HeuristicFeatureExtractor
from src.v10.ranking.lambda_rank import LambdaRankLoss, LambdaLoss
from src.v10.ranking.approx_ndcg import ApproxNDCGLoss, APFDLoss
from src.v10.ranking.ndcg_utils import compute_apfd, compute_ndcg, ranks_from_scores


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config: Dict) -> torch.device:
    """Setup compute device."""
    device_name = config.get('hardware', {}).get('device', 'cuda')
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RTPTorrentDataset(Dataset):
    """
    Dataset for RTPTorrent data.

    Each sample represents a build with multiple tests.
    For ranking, we return all tests in a build as a single sample.
    """

    def __init__(
        self,
        builds: List[Dict],
        heuristic_extractor: HeuristicFeatureExtractor,
        tokenizer: JavaTestTokenizer,
        max_tests_per_build: int = 100
    ):
        self.builds = builds
        self.heuristic_extractor = heuristic_extractor
        self.tokenizer = tokenizer
        self.max_tests_per_build = max_tests_per_build

    def __len__(self) -> int:
        return len(self.builds)

    def __getitem__(self, idx: int) -> Dict:
        build = self.builds[idx]

        # Get test info
        tests = build['tests'][:self.max_tests_per_build]
        test_ids = [t['test_id'] for t in tests]
        results = [1 if t['result'] == 'Fail' else 0 for t in tests]

        # Tokenize test names
        test_texts = [self.tokenizer.tokenize_test_name(tid) for tid in test_ids]

        # Get heuristic features
        heuristic_features = self.heuristic_extractor.extract_batch(
            test_ids, current_build=build['build_id']
        )

        # Create mask for padding
        num_tests = len(tests)
        mask = torch.ones(num_tests)

        return {
            'build_id': build['build_id'],
            'test_ids': test_ids,
            'test_texts': test_texts,
            'relevances': torch.tensor(results, dtype=torch.float32),
            'heuristic_features': heuristic_features,
            'mask': mask,
            'num_tests': num_tests
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for variable-length builds."""
    # For now, we process one build at a time
    # TODO: Implement proper padding for batching
    return batch[0]


def create_model(config: Dict, device: torch.device) -> FiloPrioriV10:
    """Create and initialize the V10 model."""
    model_config = V10Config(
        encoder_model=config['encoder'].get('model_name', 'microsoft/codebert-base'),
        encoder_dim=config['encoder'].get('output_dim', 768),
        use_co_attention=config['encoder'].get('use_co_attention', True),
        co_attention_heads=config['encoder'].get('co_attention_heads', 8),
        graph_hidden_dim=config['graph']['gat'].get('hidden_dim', 256),
        graph_num_heads=config['graph']['gat'].get('num_heads', 4),
        graph_num_layers=config['graph']['gat'].get('num_layers', 2),
        heuristic_dim=config['heuristics'].get('dim', 6),
        use_heuristics=config['heuristics'].get('enabled', True),
        initial_heuristic_weight=config['heuristics'].get('initial_weight', 0.7),
        fusion_type=config['model']['fusion'].get('type', 'hierarchical'),
        fusion_hidden_dim=config['model']['fusion'].get('hidden_dim', 256),
        dropout=config['training'].get('dropout', 0.3)
    )

    model = FiloPrioriV10(model_config)
    model = model.to(device)

    # Freeze encoder if specified
    freeze_layers = config['encoder'].get('freeze_layers', 0)
    if freeze_layers > 0:
        model.freeze_encoder(freeze_layers)

    return model


def create_loss_function(config: Dict) -> nn.Module:
    """Create the loss function based on config."""
    loss_config = config['training']['loss']
    loss_type = loss_config.get('type', 'lambda_rank')

    if loss_type == 'lambda_rank':
        return LambdaRankLoss(
            sigma=loss_config.get('sigma', 1.0),
            ndcg_at_k=loss_config.get('ndcg_at_k', None)
        )
    elif loss_type == 'lambda_loss':
        return LambdaLoss(
            sigma=loss_config.get('sigma', 1.0),
            weighting_scheme=loss_config.get('weighting', 'lambdarank')
        )
    elif loss_type == 'approx_ndcg':
        return ApproxNDCGLoss(
            temperature=loss_config.get('temperature', 1.0),
            k=loss_config.get('ndcg_at_k', None)
        )
    elif loss_type == 'apfd':
        return APFDLoss(
            temperature=loss_config.get('temperature', 1.0)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def create_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Create optimizer with different LRs for components."""
    lr_config = config['training'].get('learning_rates', {})
    base_lr = config['training'].get('learning_rate', 1e-4)
    weight_decay = config['training'].get('weight_decay', 1e-4)

    # Group parameters by component
    param_groups = []

    # Encoder (CodeBERT)
    encoder_params = [p for n, p in model.named_parameters() if 'codebert' in n]
    if encoder_params:
        param_groups.append({
            'params': encoder_params,
            'lr': lr_config.get('encoder', 1e-5)
        })

    # GAT layers
    gat_params = [p for n, p in model.named_parameters() if 'gat' in n]
    if gat_params:
        param_groups.append({
            'params': gat_params,
            'lr': lr_config.get('gat', base_lr)
        })

    # Fusion layers
    fusion_params = [p for n, p in model.named_parameters()
                    if 'fusion' in n or 'proj' in n or 'combiner' in n]
    if fusion_params:
        param_groups.append({
            'params': fusion_params,
            'lr': lr_config.get('fusion', base_lr * 10)
        })

    # Alpha parameter (if exists)
    alpha_params = [p for n, p in model.named_parameters() if 'alpha' in n.lower()]
    if alpha_params:
        param_groups.append({
            'params': alpha_params,
            'lr': lr_config.get('alpha', 1e-2)
        })

    # Any remaining parameters
    named_params = set(n for n, _ in model.named_parameters())
    grouped_params = set()
    for group in param_groups:
        for p in group['params']:
            for n, param in model.named_parameters():
                if p is param:
                    grouped_params.add(n)

    remaining = [p for n, p in model.named_parameters() if n not in grouped_params]
    if remaining:
        param_groups.append({
            'params': remaining,
            'lr': base_lr
        })

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    accumulation_steps: int = 1
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        # Move to device
        heuristic_features = batch['heuristic_features'].to(device)
        relevances = batch['relevances'].to(device)
        mask = batch['mask'].to(device)

        # Forward pass
        with autocast(enabled=scaler is not None):
            scores, _ = model(
                test_texts=batch['test_texts'],
                heuristic_features=heuristic_features.unsqueeze(0)
            )

            # Compute loss
            loss = criterion(
                scores.unsqueeze(0),
                relevances.unsqueeze(0),
                mask.unsqueeze(0)
            )
            loss = loss / accumulation_steps

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step
        if (step + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        num_batches += 1

        pbar.set_postfix({'loss': total_loss / num_batches})

    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation/test data."""
    model.eval()

    all_apfd = []
    all_ndcg = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            heuristic_features = batch['heuristic_features'].to(device)
            relevances = batch['relevances'].to(device)

            # Forward pass
            scores, _ = model(
                test_texts=batch['test_texts'],
                heuristic_features=heuristic_features.unsqueeze(0)
            )

            # Compute rankings
            rankings = ranks_from_scores(scores.unsqueeze(0))

            # Compute metrics
            apfd = compute_apfd(rankings, relevances.unsqueeze(0))
            ndcg = compute_ndcg(
                relevances[rankings[0].argsort()].unsqueeze(0),
                k=10
            )

            all_apfd.append(apfd.item())
            all_ndcg.append(ndcg.item())

    return {
        'apfd': np.mean(all_apfd),
        'apfd_std': np.std(all_apfd),
        'ndcg_at_10': np.mean(all_ndcg),
        'num_builds': len(all_apfd)
    }


def main(config_path: str):
    """Main training function."""
    # Load config
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # Setup
    device = setup_device(config)
    seed = config.get('hardware', {}).get('seed', 42)
    set_seed(seed)

    # Create output directory
    save_dir = Path(config['logging'].get('save_dir', 'results/v10'))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    logger.info("Initializing components...")

    # Heuristic extractor
    heuristic_extractor = HeuristicFeatureExtractor(
        recency_transform=config['heuristics'].get('recency_transform', 'inverse_log'),
        recent_window=config['heuristics'].get('recent_window', 5)
    )

    # Tokenizer
    tokenizer = JavaTestTokenizer()

    # TODO: Load actual data
    # For now, create placeholder
    logger.warning("Using placeholder data - implement actual data loading")
    train_builds = []
    val_builds = []

    # Create datasets
    train_dataset = RTPTorrentDataset(
        builds=train_builds,
        heuristic_extractor=heuristic_extractor,
        tokenizer=tokenizer
    )
    val_dataset = RTPTorrentDataset(
        builds=val_builds,
        heuristic_extractor=heuristic_extractor,
        tokenizer=tokenizer
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # One build per batch for ranking
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get('hardware', {}).get('num_workers', 4)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Create model
    logger.info("Creating model...")
    model = create_model(config, device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss and optimizer
    criterion = create_loss_function(config)
    optimizer = create_optimizer(model, config)

    # Learning rate scheduler
    scheduler_config = config['training'].get('scheduler', {})
    if scheduler_config.get('type') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=scheduler_config.get('min_lr', 1e-6)
        )
    else:
        scheduler = None

    # Mixed precision scaler
    scaler = GradScaler() if config['training'].get('use_amp', True) else None

    # Training loop
    logger.info("Starting training...")
    best_apfd = 0.0
    patience = config['training']['early_stopping'].get('patience', 10)
    patience_counter = 0

    for epoch in range(config['training']['epochs']):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        logger.info(f"{'='*50}")

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler,
            accumulation_steps=config['training'].get('accumulation_steps', 1)
        )
        logger.info(f"Train Loss: {train_loss:.4f}")

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        logger.info(f"Val APFD: {val_metrics['apfd']:.4f} (+/- {val_metrics['apfd_std']:.4f})")
        logger.info(f"Val NDCG@10: {val_metrics['ndcg_at_10']:.4f}")
        logger.info(f"Alpha: {model.get_alpha():.4f}")

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Early stopping
        if val_metrics['apfd'] > best_apfd:
            best_apfd = val_metrics['apfd']
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_apfd': best_apfd,
                'config': config
            }, save_dir / 'best_model.pt')
            logger.info(f"Saved best model (APFD: {best_apfd:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    logger.info(f"\nTraining complete! Best APFD: {best_apfd:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filo-Priori V10 Training')
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/experiment_v10_rtptorrent.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()

    main(args.config)
