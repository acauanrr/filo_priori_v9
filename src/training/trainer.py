"""
Training Module
Handles the training loop, optimization, and checkpointing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler, Sampler
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau
from typing import Dict, Optional, Tuple, List, Iterator
import logging
import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class MultiFieldDataset(Dataset):
    """Dataset that handles multi-field embeddings"""

    def __init__(
        self,
        field_embeddings: Dict[str, np.ndarray],  # Dict[field_name, array]
        labels: np.ndarray,
        sample_indices: np.ndarray,
        groups: Optional[np.ndarray] = None,
        field_order: List[str] = None
    ):
        """
        Args:
            field_embeddings: Dict mapping field_name → embeddings [N, field_dim]
            labels: Label array [N]
            sample_indices: Sample indices [N]
            groups: Optional group IDs [N]
            field_order: Order of fields (e.g., ['summary', 'steps', 'commits', 'CR'])
        """
        self.field_embeddings = field_embeddings
        self.labels = labels
        self.sample_indices = sample_indices
        self.groups = groups

        # Determine field order
        if field_order is None:
            self.field_order = sorted(field_embeddings.keys())
        else:
            self.field_order = field_order

        # Convert to tensors
        self.field_tensors = {
            name: torch.tensor(field_embeddings[name], dtype=torch.float32)
            for name in self.field_order
        }
        self.labels_tensor = torch.tensor(labels, dtype=torch.long)
        self.indices_tensor = torch.tensor(sample_indices, dtype=torch.long)

        if groups is not None:
            self.groups_tensor = torch.tensor(groups, dtype=torch.long)
        else:
            self.groups_tensor = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return list of field embeddings in specified order
        field_list = [self.field_tensors[name][idx] for name in self.field_order]

        if self.groups_tensor is not None:
            return field_list, self.labels_tensor[idx], self.groups_tensor[idx], self.indices_tensor[idx]
        else:
            return field_list, self.labels_tensor[idx], self.indices_tensor[idx]


class GroupedBatchSampler(Sampler):
    """
    Batch sampler that groups samples by Build_ID to enable pairwise ranking loss.

    Ensures each batch contains samples from the same build(s), so that
    pairwise ranking loss can find (Fail, Pass) pairs within builds.

    Strategy:
    - Group all indices by their Build_ID
    - Shuffle build order at each epoch
    - Yield batches containing entire build(s) or portions of large builds
    """

    def __init__(
        self,
        groups: np.ndarray,  # Build_ID for each sample
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Args:
            groups: Array of group IDs (Build_ID) for each sample [N]
            batch_size: Target batch size
            drop_last: Drop incomplete batches
            shuffle: Shuffle build order at each epoch
            seed: Random seed
        """
        self.groups = groups
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

        # Group indices by Build_ID
        self.build_to_indices = defaultdict(list)
        for idx, build_id in enumerate(groups):
            self.build_to_indices[build_id].append(idx)

        self.build_ids = list(self.build_to_indices.keys())
        self.num_builds = len(self.build_ids)

        logger.info(f"GroupedBatchSampler: {len(groups)} samples across {self.num_builds} builds")
        logger.info(f"  Avg samples/build: {len(groups)/self.num_builds:.1f}")
        logger.info(f"  Target batch size: {batch_size}")

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle build order if requested
        if self.shuffle:
            build_order = self.rng.permutation(self.build_ids).tolist()
        else:
            build_order = self.build_ids.copy()

        batch = []

        for build_id in build_order:
            build_indices = self.build_to_indices[build_id]

            # Shuffle indices within build if requested
            if self.shuffle:
                build_indices = self.rng.permutation(build_indices).tolist()

            # Add build samples to current batch
            batch.extend(build_indices)

            # Yield batch if it reaches target size
            while len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

        # Handle remaining samples
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total_samples = len(self.groups)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


class Trainer:
    """Handles model training and evaluation"""

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        train_data: Dict,
        val_data: Dict,
        graph_data: Dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.graph_data = graph_data
        self.device = device
        self.training_config = self.config['training']

        # Normalize optional configuration blocks for backward compatibility
        self.early_stopping_config = self._resolve_early_stopping_config()
        self.save_every_n_epochs = self._resolve_save_every_n_epochs()
        self.gradient_clip_value = self._resolve_gradient_clip_value()
        self.temperature = 1.0  # probability calibration (temperature scaling)

        # Move model to device
        self.model = self.model.to(device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Setup loss function (imported separately)
        self.criterion = None  # Will be set externally

        # Training state
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.patience_counter = 0
        self.best_threshold = 0.5  # Default threshold for binary classification

        # Check if using multi-field embeddings
        self.use_multi_field = train_data.get('is_multi_field', False)
        if self.use_multi_field:
            # Get field order from config
            fields = config['embedding'].get('fields', [])
            self.field_order = [f['name'] for f in fields]
            logger.info(f"Using multi-field embeddings with fields: {self.field_order}")
        else:
            self.field_order = None

        # Prepare group encoder and data loaders
        self.group_encoder = self._build_group_encoder(train_data, val_data)
        self.train_loader = self._create_dataloader(train_data, shuffle=True)
        self.val_loader = self._create_dataloader(val_data, shuffle=False)

        # Store full graph structure (as numpy for flexibility)
        self.full_edge_index = graph_data['edge_index']
        self.full_edge_weights = graph_data['edge_weights']

        # Torch versions for inducing subgraphs on-device
        try:
            self.full_edge_index_torch = torch.tensor(self.full_edge_index, dtype=torch.long, device=self.device)
        except Exception:
            # In case already tensor
            self.full_edge_index_torch = self.full_edge_index.to(self.device) if hasattr(self.full_edge_index, 'to') else torch.zeros((2, 0), dtype=torch.long, device=self.device)
        if self.full_edge_weights is None:
            self.full_edge_weights_torch = None
        else:
            try:
                self.full_edge_weights_torch = torch.tensor(self.full_edge_weights, dtype=torch.float32, device=self.device)
            except Exception:
                self.full_edge_weights_torch = self.full_edge_weights.to(self.device) if hasattr(self.full_edge_weights, 'to') else None

        # Store full training embeddings for graph lookups
        if self.use_multi_field:
            # For multi-field, store as dict of tensors
            field_embeddings = train_data['field_embeddings']
            self.full_train_embeddings = {
                name: torch.tensor(field_embeddings[name], dtype=torch.float32, device=device)
                for name in self.field_order
            }
        else:
            # Original single embedding
            self.full_train_embeddings = torch.tensor(
                train_data['embeddings'],
                dtype=torch.float32,
                device=device
            )

    def _resolve_early_stopping_config(self) -> Dict:
        """
        Build an early stopping configuration with sensible defaults so that
        legacy configs (that only define patience) still work.
        """
        default_patience = self.config['training'].get('early_stopping_patience', 10)
        default_monitor = self.config['training'].get('primary_metric', 'auprc_macro')
        default_mode = self.config['training'].get('early_stopping_mode', 'max')

        early_cfg = self.config['training'].get('early_stopping')
        if early_cfg is None:
            return {
                'patience': default_patience,
                'monitor': default_monitor,
                'mode': default_mode
            }

        # Work on a shallow copy to avoid mutating the original config
        normalized = dict(early_cfg)
        normalized.setdefault('patience', default_patience)
        normalized.setdefault('monitor', default_monitor)
        normalized.setdefault('mode', default_mode)
        return normalized

    def _resolve_save_every_n_epochs(self) -> Optional[int]:
        """
        Normalize checkpoint frequency, returning None if not configured
        or invalid (<= 0).
        """
        save_every = self.config['training'].get('save_every_n_epochs')
        if save_every is None:
            return None
        try:
            save_every_int = int(save_every)
            if save_every_int <= 0:
                logger.warning("save_every_n_epochs <= 0; periodic checkpoints disabled")
                return None
            return save_every_int
        except (TypeError, ValueError):
            logger.warning("Invalid save_every_n_epochs value; periodic checkpoints disabled")
            return None

    def _resolve_gradient_clip_value(self) -> Optional[float]:
        """
        Support both legacy 'clip_grad_norm' and newer 'gradient_clip' keys.
        """
        clip_value = self.config['training'].get('gradient_clip')
        if clip_value is None:
            clip_value = self.config['training'].get('clip_grad_norm')
        if clip_value is None:
            return None

        try:
            clip_float = float(clip_value)
            if clip_float <= 0:
                logger.warning("Non-positive gradient clip value ignored")
                return None
            return clip_float
        except (TypeError, ValueError):
            logger.warning("Invalid gradient clip value; clipping disabled")
            return None

    def _create_optimizer(self):
        """Create optimizer based on config"""
        training_config = self.training_config
        optimizer_type = training_config['optimizer']
        lr = float(training_config['learning_rate'])
        weight_decay = float(training_config['weight_decay'])

        if optimizer_type == 'adamw':
            return AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adam':
            return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        training_config = self.training_config
        scheduler_type = training_config['scheduler']
        num_epochs = training_config['num_epochs']

        if scheduler_type == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif scheduler_type == 'linear':
            return LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(self.optimizer, mode='max', patience=5)
        else:
            return None

    def _build_group_encoder(self, train_data: Dict, val_data: Dict) -> Dict[str, int]:
        """Map Build_ID strings to integer group ids across train/val if available."""
        all_ids = []
        for d in (train_data, val_data):
            if 'build_ids' in d and d['build_ids'] is not None:
                all_ids.extend([str(x) for x in d['build_ids']])
        mapping = {}
        for bid in all_ids:
            if bid not in mapping:
                mapping[bid] = len(mapping)
        return mapping

    def _create_dataloader(self, data: Dict, shuffle: bool = False) -> DataLoader:
        """Create PyTorch DataLoader with optional class-balanced sampling"""

        # Check if multi-field or single embedding
        is_multi_field = data.get('is_multi_field', False)

        labels = torch.tensor(data['labels'], dtype=torch.long)

        # Optional groups (Build_ID → int)
        groups_tensor = None
        if 'build_ids' in data and data['build_ids'] is not None:
            gids = [self.group_encoder.get(str(x), -1) for x in data['build_ids']]
            groups_tensor = torch.tensor(gids, dtype=torch.long)

        # Create dataset based on embedding type
        if is_multi_field:
            # Multi-field embeddings
            field_embeddings = data['field_embeddings']
            sample_indices = np.arange(len(labels))

            # Safety check
            first_field = list(field_embeddings.values())[0]
            n_e, n_l = len(first_field), len(labels)
            if groups_tensor is not None and len(groups_tensor) != n_l:
                logger.warning(
                    f"Size mismatch detected for multi-field dataset creation: "
                    f"embeddings={n_e}, labels={n_l}, groups={len(groups_tensor)}. "
                    f"Ignoring groups to proceed without ranking loss."
                )
                groups_tensor = None

            groups_np = groups_tensor.numpy() if groups_tensor is not None else None

            dataset = MultiFieldDataset(
                field_embeddings=field_embeddings,
                labels=labels.numpy(),
                sample_indices=sample_indices,
                groups=groups_np,
                field_order=self.field_order
            )
        else:
            # Single embedding (original)
            embeddings = torch.tensor(data['embeddings'], dtype=torch.float32)

            # Safety: if group ids length does not match, ignore groups for this split
            if groups_tensor is not None:
                n_e, n_l, n_g = embeddings.size(0), labels.size(0), groups_tensor.size(0)
                if not (n_e == n_l == n_g):
                    logger.warning(
                        f"Size mismatch detected for dataset creation: "
                        f"embeddings={n_e}, labels={n_l}, groups={n_g}. "
                        f"Ignoring groups to proceed without ranking loss."
                    )
                    groups_tensor = None

            # Always include sample indices for batch-to-global mapping (for global graph usage)
            sample_indices = torch.arange(embeddings.size(0), dtype=torch.long)
            if groups_tensor is None:
                dataset = TensorDataset(embeddings, labels, sample_indices)
            else:
                dataset = TensorDataset(embeddings, labels, groups_tensor, sample_indices)

        # Check if ranking is enabled and we should use GroupedBatchSampler
        ranking_cfg = self.config['training'].get('ranking', {})
        ranking_enabled = ranking_cfg.get('enabled', False)
        use_grouped_sampler = ranking_cfg.get('use_grouped_sampler', False)

        # Use GroupedBatchSampler for ranking if enabled and groups available
        if shuffle and ranking_enabled and use_grouped_sampler and 'build_ids' in data and data['build_ids'] is not None:
            logger.info("🔄 Using GroupedBatchSampler (ranking-aware batching)")
            batch_sampler = GroupedBatchSampler(
                groups=np.array(data['build_ids']),
                batch_size=self.config['training']['batch_size'],
                drop_last=False,
                shuffle=True,
                seed=self.config['data']['random_seed']
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=0,
                pin_memory=self.config['hardware'].get('pin_memory', True)
            )
        # Use class-balanced sampling for training data (original behavior)
        elif shuffle and self.training_config.get('use_balanced_sampling', False):
            logger.info("🔄 Using WeightedRandomSampler (class-balanced sampling)")

            # Compute sample weights: inverse of class frequency
            class_counts = torch.bincount(labels)
            class_weights_tensor = 1.0 / class_counts.float()
            sample_weights = class_weights_tensor[labels]

            # Create weighted sampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(dataset),
                replacement=True  # Allow oversampling
            )

            logger.info(f"Class counts: {class_counts.tolist()}")
            logger.info(f"Class weights: {class_weights_tensor.tolist()}")

            return DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                sampler=sampler,  # Use sampler instead of shuffle
                num_workers=0,  # Set to 0 to avoid tokenizer fork warnings
                pin_memory=self.config['hardware'].get('pin_memory', True)
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=shuffle,
                num_workers=0,  # Set to 0 to avoid tokenizer fork warnings
                pin_memory=self.config['hardware'].get('pin_memory', True)
            )

    def _rebuild_train_loader_with_weights(self, sample_weights: torch.Tensor):
        """Recreate train DataLoader with sampler based on ranking config."""
        data = self.train_data
        embeddings = torch.tensor(data['embeddings'], dtype=torch.float32)
        labels = torch.tensor(data['labels'], dtype=torch.long)
        groups_tensor = None
        if 'build_ids' in data and data['build_ids'] is not None:
            gids = [self.group_encoder.get(str(x), -1) for x in data['build_ids']]
            groups_tensor = torch.tensor(gids, dtype=torch.long)

        # Safety: drop groups if lengths mismatch
        if groups_tensor is not None:
            n_e, n_l, n_g = embeddings.size(0), labels.size(0), groups_tensor.size(0)
            if not (n_e == n_l == n_g):
                logger.warning(
                    f"Size mismatch rebuilding train loader: "
                    f"embeddings={n_e}, labels={n_l}, groups={n_g}. "
                    f"Ignoring groups."
                )
                groups_tensor = None

        # Always include sample indices for batch-to-global mapping (for global graph usage)
        sample_indices = torch.arange(embeddings.size(0), dtype=torch.long)
        if groups_tensor is None:
            dataset = TensorDataset(embeddings, labels, sample_indices)
        else:
            dataset = TensorDataset(embeddings, labels, groups_tensor, sample_indices)

        # Choose sampler based on ranking config
        ranking_cfg = self.config['training'].get('ranking', {})
        ranking_enabled = ranking_cfg.get('enabled', False)
        use_grouped_sampler = ranking_cfg.get('use_grouped_sampler', False)

        if ranking_enabled and use_grouped_sampler and groups_tensor is not None:
            # Use GroupedBatchSampler for ranking (groups samples by Build_ID)
            logger.info("🔄 Using GroupedBatchSampler (ranking-aware batching)")
            batch_sampler = GroupedBatchSampler(
                groups=data['build_ids'],
                batch_size=self.config['training']['batch_size'],
                drop_last=False,
                shuffle=True,
                seed=self.config['data']['random_seed']
            )
            self.train_loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=0,
                pin_memory=self.config['hardware'].get('pin_memory', True)
            )
        else:
            # Use WeightedRandomSampler (class balancing, but destroys build structure)
            logger.info("🔄 Using WeightedRandomSampler (class balancing)")
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(dataset),
                replacement=True
            )
            self.train_loader = DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                sampler=sampler,
                num_workers=0,
                pin_memory=self.config['hardware'].get('pin_memory', True)
            )

    def _build_batch_knn_graph(self, embeddings: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build k-NN graph for a batch of embeddings using COSINE distance

        Args:
            embeddings: Batch embeddings [batch_size, embedding_dim]
            k: Number of neighbors

        Returns:
            edge_index [2, num_edges], edge_weights [num_edges]
        """
        batch_size = embeddings.size(0)

        # Compute pairwise COSINE distances
        # Normalize embeddings to unit vectors
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Cosine similarity: dot product of normalized vectors
        similarity = torch.mm(embeddings_norm, embeddings_norm.t())

        # Cosine distance = 1 - cosine similarity
        distances = 1.0 - similarity

        # Find k nearest neighbors for each node (excluding self)
        # Add small value to diagonal to avoid selecting self
        distances = distances + torch.eye(batch_size, device=embeddings.device) * 1e10

        # Get k smallest distances and their indices
        k_actual = min(k, batch_size - 1)  # Can't have more neighbors than other nodes
        _, knn_indices = torch.topk(distances, k_actual, dim=1, largest=False)

        # Build edge list
        edge_index_list = []
        edge_weight_list = []

        for i in range(batch_size):
            for j in range(k_actual):
                neighbor_idx = knn_indices[i, j].item()
                edge_index_list.append([i, neighbor_idx])

                # Convert distance to similarity weight
                dist = distances[i, neighbor_idx].item()
                weight = 1.0 / (1.0 + dist)
                edge_weight_list.append(weight)

        if len(edge_index_list) == 0:
            # Fallback: create empty graph
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=embeddings.device)
            edge_weights = torch.zeros(0, dtype=torch.float32, device=embeddings.device)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long, device=embeddings.device).t()
            edge_weights = torch.tensor(edge_weight_list, dtype=torch.float32, device=embeddings.device)

        return edge_index, edge_weights

    def _build_edges_from_global_graph(self, batch_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Induce a subgraph for the current batch from the full (precomputed) graph.
        """
        device = self.device
        if self.full_edge_index_torch is None or self.full_edge_index_torch.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=device), torch.zeros(0, dtype=torch.float32, device=device)

        # Map global -> local positions
        local_map = {int(g): i for i, g in enumerate(batch_indices.tolist())}
        src = self.full_edge_index_torch[0].tolist()
        dst = self.full_edge_index_torch[1].tolist()

        sub_src_local = []
        sub_dst_local = []
        sub_weights = []

        use_weights = self.full_edge_weights_torch is not None
        for e_idx, (s, d) in enumerate(zip(src, dst)):
            if s in local_map and d in local_map:
                sub_src_local.append(local_map[s])
                sub_dst_local.append(local_map[d])
                if use_weights:
                    sub_weights.append(float(self.full_edge_weights_torch[e_idx].item()))

        if len(sub_src_local) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_weights = torch.zeros(0, dtype=torch.float32, device=device)
        else:
            edge_index = torch.tensor([sub_src_local, sub_dst_local], dtype=torch.long, device=device)
            edge_weights = torch.tensor(sub_weights, dtype=torch.float32, device=device) if use_weights else torch.zeros(len(sub_src_local), dtype=torch.float32, device=device)

        return edge_index, edge_weights

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        k_neighbors = self.config['model']['structural_stream']['num_neighbors']
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        ranking_cfg = self.config['training'].get('ranking', {})
        ranking_enabled = ranking_cfg.get('enabled', False)

        for batch_idx, batch in enumerate(progress_bar):
            # Handle both multi-field and single embedding datasets
            # Multi-field: first element is list of field tensors
            # Single: first element is single tensor
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                embeddings, labels, groups, sample_idx = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                # No groups in this split
                embeddings, labels, sample_idx = batch
                groups = None
            else:
                embeddings, labels = batch
                groups = None
                sample_idx = torch.arange(embeddings[0].size(0) if isinstance(embeddings, list) else embeddings.size(0), dtype=torch.long)

            # Move to device
            if isinstance(embeddings, list):
                # Multi-field: move each field tensor to device
                embeddings = [emb.to(self.device) for emb in embeddings]
            else:
                # Single embedding
                embeddings = embeddings.to(self.device)

            labels = labels.to(self.device)
            groups_tensor = groups.to(self.device) if groups is not None else None

            # Build edges based on configured graph mode
            graph_mode = self.config['training'].get('graph_mode', 'batch_knn')
            if graph_mode == 'global_rewired':
                edge_index, edge_weights = self._build_edges_from_global_graph(sample_idx.cpu())
            else:
                # For multi-field, we need to fuse first for k-NN graph building
                if isinstance(embeddings, list):
                    # Apply field fusion to get single embedding for graph building
                    embeddings_for_graph = self.model.field_fusion(embeddings)
                else:
                    embeddings_for_graph = embeddings
                edge_index, edge_weights = self._build_batch_knn_graph(embeddings_for_graph, k=k_neighbors)

            # Forward pass
            logits = self.model(
                embeddings=embeddings,
                edge_index=edge_index,
                edge_weights=edge_weights
            )

            # Compute loss
            loss = self.criterion(logits, labels)

            # Ranking-aware pairwise loss within builds (optional)
            # CURRICULUM LEARNING: Start ranking loss only after model learns basic classification
            if ranking_enabled and groups_tensor is not None and self.config['model']['classifier']['num_classes'] == 2:
                start_epoch = ranking_cfg.get('start_epoch', 0)
                ramp_epochs = ranking_cfg.get('ramp_epochs', 1)

                # Calculate dynamic ranking weight with curriculum
                rank_weight = 0.0
                if self.current_epoch >= start_epoch:
                    # Linear ramp from 0 to full weight
                    ramp_progress = min(1.0, (self.current_epoch - start_epoch) / max(1, ramp_epochs))
                    rank_weight = float(ranking_cfg.get('weight', 0.2)) * ramp_progress

                if rank_weight > 0:
                    rank_loss = self._pairwise_ranking_loss(logits, labels, groups_tensor)
                    loss = loss + rank_weight * rank_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100.0 * correct / total
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def _pairwise_ranking_loss(self, logits: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        """Pairwise ranking loss within builds with hard negative mining.

        - Score: configurable as P(Fail) or logit(Fail)
        - Loss types:
            * 'logistic'  → softplus(-(s_i - s_j))  (RankNet-style)
            * 'margin'/'hinge' → relu(margin - (s_i - s_j))
        - Mining: selects hardest negatives (Pass) per build by highest score
                  and pairs each Fail with top-K (or top-p%) Pass.

        Config (training.ranking):
            enabled: bool
            weight: float
            loss_type: 'logistic' | 'margin' | 'hinge'
            score_type: 'logit' | 'prob'          (default: 'logit')
            margin: float                          (default depends on score_type)
            hard_negative_top_k: int               (default: 5)
            hard_negative_percent: float in (0,1]  (default: 0.2 if top_k not set)
            max_pairs_per_build: int               (default: 50)
        """
        cfg = self.config['training'].get('ranking', {})
        loss_type = str(cfg.get('loss_type', 'logistic')).lower()
        score_type = str(cfg.get('score_type', 'logit')).lower()
        max_pairs = int(cfg.get('max_pairs_per_build', 50))
        # Mining params
        top_k = int(cfg.get('hard_negative_top_k', 5))
        percent = float(cfg.get('hard_negative_percent', 0.2))
        percent = max(0.0, min(1.0, percent))

        # Choose score: logit(Fail) is more stable for margins
        if score_type == 'prob':
            scores = torch.softmax(logits, dim=1)[:, 0]
            default_margin = 0.05
        else:
            scores = logits[:, 0]
            default_margin = 0.5

        margin = float(cfg.get('margin', default_margin))

        loss_terms = []
        unique_groups = groups.unique()
        for g in unique_groups:
            mask = groups == g
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue

            lbl = labels[idx]
            fail_idx = idx[lbl == 0]
            pass_idx = idx[lbl == 1]
            if fail_idx.numel() == 0 or pass_idx.numel() == 0:
                continue

            # Hard negative mining: select hardest Pass within build
            pass_scores = scores[pass_idx]
            if top_k is not None and top_k > 0:
                k = min(int(top_k), int(pass_idx.numel()))
            else:
                k = max(1, int(torch.ceil(torch.tensor(percent * float(pass_idx.numel()))).item()))
                k = min(k, int(pass_idx.numel()))

            if k <= 0:
                continue

            # Top-k by score (highest P/logit(Fail) → hardest negatives)
            topk_vals, topk_indices_local = torch.topk(pass_scores, k=k, largest=True)
            hard_pass_idx = pass_idx[topk_indices_local]

            # Pair every Fail with each hard negative Pass
            num_pairs_total = int(fail_idx.numel() * hard_pass_idx.numel())
            if num_pairs_total == 0:
                continue

            if num_pairs_total <= max_pairs:
                fi = fail_idx.repeat_interleave(hard_pass_idx.numel())
                pj = hard_pass_idx.repeat(fail_idx.numel())
            else:
                # Random subsample pairs to respect cap
                # Sample fail and pass indices independently, forming max_pairs pairs
                fi = fail_idx[torch.randint(0, fail_idx.numel(), (max_pairs,), device=labels.device)]
                pj = hard_pass_idx[torch.randint(0, hard_pass_idx.numel(), (max_pairs,), device=labels.device)]

            s_i = scores[fi]
            s_j = scores[pj]

            if loss_type in ("margin", "hinge"):
                # Hinge margin: max(0, margin - (s_i - s_j))
                loss_pair = torch.relu(margin - (s_i - s_j))
            else:
                # Logistic (RankNet)
                loss_pair = torch.nn.functional.softplus(-(s_i - s_j))  # log(1+exp(-(s_i - s_j)))

            loss_terms.append(loss_pair.mean())

        if not loss_terms:
            return torch.tensor(0.0, device=labels.device)
        return torch.stack(loss_terms).mean()

    def _optimize_threshold(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        """
        Find optimal classification threshold for binary classification
        Optimizes F1 score of FAIL class (minority class) on validation set

        CRITICAL FIX (Exp 014):
        - Previous: Optimized F1-macro (equal weight to both classes)
        - Now: Optimizes F1 of Fail class (class 0, minority)
        - Justification: APFD depends on correctly ranking Fail tests

        Args:
            probabilities: Predicted probabilities [n_samples, num_classes]
            labels: True labels [n_samples]

        Returns:
            Optimal threshold value
        """
        if self.config['model']['classifier']['num_classes'] != 2:
            # Only for binary classification
            return 0.5

        # Extract positive class probability (Pass = class 1)
        pos_probs = probabilities[:, 1]

        # Get threshold search range from config (default [0.1, 0.9])
        threshold_range = self.config['training'].get('threshold_search_range', [0.1, 0.9])
        min_thresh, max_thresh = threshold_range[0], threshold_range[1]

        # Try different thresholds
        thresholds = np.arange(min_thresh, max_thresh + 0.05, 0.05)
        best_f1_fail = 0.0
        best_thresh = 0.5

        from sklearn.metrics import f1_score

        for thresh in thresholds:
            predictions = (pos_probs >= thresh).astype(int)

            # ✅ CRITICAL FIX: Optimize F1 of Fail class (pos_label=0), not macro
            # This maximizes detection of Fail tests, which is critical for APFD
            f1_fail = f1_score(labels, predictions, pos_label=0, zero_division=0)

            if f1_fail > best_f1_fail:
                best_f1_fail = f1_fail
                best_thresh = thresh

        logger.info(f"Optimal threshold: {best_thresh:.3f} (F1 Fail class: {best_f1_fail:.4f})")
        return best_thresh

    def _check_prediction_collapse(self, predictions: np.ndarray, labels: np.ndarray) -> bool:
        """
        Check if model has collapsed to predicting only one class.
        This is a critical failure mode for imbalanced datasets.

        Args:
            predictions: Predicted labels [n_samples]
            labels: True labels [n_samples]

        Returns:
            True if collapse detected, False otherwise
        """
        unique_predictions = np.unique(predictions)
        unique_labels = np.unique(labels)

        if len(unique_predictions) == 1:
            predicted_class = unique_predictions[0]
            num_classes = len(unique_labels)
            total_predictions = len(predictions)

            logger.error("!" * 80)
            logger.error("⚠️  PREDICTION COLLAPSE DETECTED!")
            logger.error(f"⚠️  Model is predicting ONLY class {predicted_class}")
            logger.error(f"⚠️  Total predictions: {total_predictions}, All predict class {predicted_class}")
            logger.error(f"⚠️  This is a trivial classifier - model is NOT learning!")
            logger.error("!" * 80)

            # Show class distribution in ground truth
            for class_idx in unique_labels:
                count = np.sum(labels == class_idx)
                percentage = 100.0 * count / len(labels)
                is_predicted = "✓ PREDICTED" if class_idx == predicted_class else "✗ NOT PREDICTED"
                logger.error(f"  Class {class_idx}: {count} samples ({percentage:.1f}%) - {is_predicted}")

            logger.error("!" * 80)
            return True

        elif len(unique_predictions) < len(unique_labels):
            missing_classes = set(unique_labels) - set(unique_predictions)
            logger.warning("⚠️  Partial prediction collapse detected!")
            logger.warning(f"⚠️  Model is NOT predicting classes: {missing_classes}")
            logger.warning(f"⚠️  Only predicting classes: {set(unique_predictions)}")

        return False

    @torch.no_grad()
    def validate(self, optimize_threshold: bool = False) -> Dict[str, float]:
        """
        Validate the model

        Args:
            optimize_threshold: Whether to find optimal classification threshold

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_probabilities = []
        all_labels = []

        k_neighbors = self.config['model']['structural_stream']['num_neighbors']

        all_groups = []
        for batch in tqdm(self.val_loader, desc="Validating"):
            # Our DataLoader returns (E, L, idx) or (E, L, G, idx)
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                embeddings, labels, groups, _indices = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                embeddings, labels, _indices = batch
                groups = None
            else:
                embeddings, labels = batch
                groups = None

            # Move to device (handle multi-field)
            if isinstance(embeddings, list):
                embeddings = [emb.to(self.device) for emb in embeddings]
            else:
                embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            # Build batch-specific k-NN graph
            if isinstance(embeddings, list):
                # Multi-field: fuse first for k-NN
                embeddings_for_graph = self.model.field_fusion(embeddings)
            else:
                embeddings_for_graph = embeddings
            edge_index, edge_weights = self._build_batch_knn_graph(embeddings_for_graph, k=k_neighbors)

            # Forward pass
            logits = self.model(
                embeddings=embeddings,
                edge_index=edge_index,
                edge_weights=edge_weights
            )

            # Compute loss
            loss = self.criterion(logits, labels)

            # Get probabilities and predictions
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if groups is not None:
                all_groups.extend(groups.cpu().numpy().tolist())

        # Convert to numpy arrays
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        # Optimize threshold if requested
        if optimize_threshold:
            optimal_threshold = self._optimize_threshold(all_probabilities, all_labels)
            # Re-compute predictions with optimal threshold
            if self.config['model']['classifier']['num_classes'] == 2:
                all_predictions = (all_probabilities[:, 1] >= optimal_threshold).astype(int)
                # Recompute accuracy with new threshold
                correct = np.sum(all_predictions == all_labels)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        result = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'labels': all_labels
        }

        # Ranking metrics if groups available and binary classification
        if all_groups and self.config['model']['classifier']['num_classes'] == 2:
            try:
                from ..evaluation.metrics import compute_ranking_metrics_by_build
                ranking_cfg = self.config['training'].get('ranking', {}).get('metrics', {})
                ks = ranking_cfg.get('ndcg_ks', [5, 10])
                percents = ranking_cfg.get('recall_at_percent', [0.1])
                rank_metrics = compute_ranking_metrics_by_build(
                    np.array(all_probabilities),
                    np.array(all_labels),
                    np.array(all_groups),
                    ks=ks,
                    percents=percents
                )
                for k, v in rank_metrics.items():
                    logger.info(f"Val {k}: {v:.4f}")
                result.update(rank_metrics)
            except Exception as e:
                logger.warning(f"Ranking metric computation failed: {e}")

        return result

    def train(self, num_epochs: Optional[int] = None):
        """
        Full training loop

        Args:
            num_epochs: Number of epochs (defaults to config)
        """
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']

        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Optional Hard-Negative Mining: update sampler based on current model
            hnm_cfg = self.config['training'].get('hard_negative_mining', {})
            if hnm_cfg.get('enabled', False) and epoch > 0 and (epoch % int(hnm_cfg.get('update_interval_epochs', 1)) == 0):
                try:
                    weights = self._compute_hard_negative_weights()
                    self._rebuild_train_loader_with_weights(weights)
                    logger.info("Updated train sampler with hard-negative mining weights")
                except Exception as e:
                    logger.warning(f"Hard-negative mining update failed: {e}")

            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                       f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Train Acc: {train_metrics['accuracy']:.2f}%")

            # Validate (with threshold optimization on best epochs)
            optimize_thresh = self.config['training'].get('optimize_threshold', False)
            val_metrics = self.validate(optimize_threshold=optimize_thresh)
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Acc: {val_metrics['accuracy']:.2f}%")

            # Compute additional metrics (including AUPRC)
            from ..evaluation.metrics import compute_metrics
            detailed_metrics = compute_metrics(
                val_metrics['predictions'],
                val_metrics['labels'],
                num_classes=self.config['model']['classifier']['num_classes'],
                probabilities=val_metrics['probabilities']
            )

            # Check for prediction collapse (critical failure mode)
            collapse_detected = self._check_prediction_collapse(
                val_metrics['predictions'],
                val_metrics['labels']
            )

            # Get monitored metric for early stopping
            monitor_metric_name = self.early_stopping_config.get('monitor', 'auprc_macro')

            if monitor_metric_name == 'auprc_macro':
                current_metric = detailed_metrics.get('auprc_macro', 0.0)
                logger.info(f"Val AUPRC (Macro): {current_metric:.4f}, "
                           f"Val AUPRC (Weighted): {detailed_metrics.get('auprc_weighted', 0.0):.4f}")
            else:
                current_metric = detailed_metrics.get(monitor_metric_name, 0.0)

            # Enhanced metrics logging: precision, recall, F1
            logger.info(f"Val F1 (Macro): {detailed_metrics['f1_macro']:.4f}, "
                       f"Val F1 (Weighted): {detailed_metrics['f1_weighted']:.4f}")
            logger.info(f"Val Precision (Macro): {detailed_metrics['precision_macro']:.4f}, "
                       f"Val Recall (Macro): {detailed_metrics['recall_macro']:.4f}")

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(current_metric)
                else:
                    self.scheduler.step()

            # Check for improvement
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0

                # Update best threshold for binary classification
                if self.config['model']['classifier']['num_classes'] == 2 and optimize_thresh:
                    self.best_threshold = self._optimize_threshold(
                        val_metrics['probabilities'],
                        val_metrics['labels']
                    )

                # Optional: calibrate probabilities on validation set (temperature scaling)
                if self.training_config.get('calibrate_probabilities', False):
                    try:
                        self.temperature = float(self._calibrate_temperature())
                        logger.info(f"Calibrated temperature: {self.temperature:.4f}")
                    except Exception as e:
                        logger.warning(f"Temperature calibration failed: {e}")

                # Save best model (including threshold and temperature)
                self.save_checkpoint('best_model.pt')
                logger.info(f"New best model saved! {monitor_metric_name}: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1

            # Early stopping
            early_stop_patience = self.early_stopping_config.get('patience', 10)
            if self.patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Periodic checkpoint
            if self.save_every_n_epochs and self.save_every_n_epochs > 0:
                if (epoch + 1) % self.save_every_n_epochs == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

        monitor_metric_name = self.early_stopping_config.get('monitor', 'auprc_macro')
        logger.info("Training complete!")
        logger.info(f"Best {monitor_metric_name}: {self.best_metric:.4f}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config['data']['output_dir'], 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, filename)

        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'best_threshold': self.best_threshold,
            'temperature': self.temperature,
            'config': self.config
        }, checkpoint_path)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint_path = os.path.join(
            self.config['data']['output_dir'],
            'checkpoints',
            filename
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.best_threshold = checkpoint.get('best_threshold', 0.5)
        self.temperature = float(checkpoint.get('temperature', 1.0))

        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Resumed from epoch {self.current_epoch}, best metric: {self.best_metric:.4f}")
        if self.config['model']['classifier']['num_classes'] == 2:
            logger.info(f"Best threshold: {self.best_threshold:.3f}")
            logger.info(f"Temperature: {self.temperature:.4f}")

    @torch.no_grad()
    def _compute_hard_negative_weights(self) -> torch.Tensor:
        """
        Compute per-sample weights emphasizing hard negatives (Pass labeled as 1
        with high P(Fail)). Returns a tensor of shape [N_train].
        """
        cfg = self.config['training'].get('hard_negative_mining', {})
        top_k_percent = float(cfg.get('top_k_percent', 0.2))
        pass_boost = float(cfg.get('pass_boost', 2.0))
        min_w = float(cfg.get('min_weight', 0.1))
        max_w = float(cfg.get('max_weight', 10.0))

        # Build a non-shuffled loader over train set to get probabilities in order
        data = self.train_data
        embeddings = torch.tensor(data['embeddings'], dtype=torch.float32, device=self.device)
        labels = torch.tensor(data['labels'], dtype=torch.long, device=self.device)
        batch_size = self.config['training']['batch_size']

        probs_fail_list = []
        for i in range(0, embeddings.size(0), batch_size):
            batch_emb = embeddings[i:i+batch_size]
            edge_index, edge_weights = self._build_batch_knn_graph(batch_emb, k=self.config['model']['structural_stream']['num_neighbors'])
            logits = self.model(batch_emb, edge_index, edge_weights)
            probs = torch.softmax(logits, dim=1)
            probs_fail_list.append(probs[:, 0].detach().cpu())
        probs_fail = torch.cat(probs_fail_list, dim=0)  # on CPU

        # Base weights: inverse class frequency
        labels_cpu = labels.detach().cpu()
        class_counts = torch.bincount(labels_cpu, minlength=2).float()
        inv = 1.0 / torch.clamp(class_counts, min=1.0)
        base_weights = inv[labels_cpu]

        # Identify hard negatives among Pass (label=1)
        pass_mask = labels_cpu == 1
        num_pass = int(pass_mask.sum().item())
        if num_pass > 0:
            k = max(1, int(top_k_percent * num_pass))
            pass_indices = torch.nonzero(pass_mask, as_tuple=False).squeeze(1)
            pass_scores = probs_fail[pass_indices]
            # top-k by prob_fail
            topk = torch.topk(pass_scores, k=k, largest=True).indices
            hard_pass_idx = pass_indices[topk]
            weights = base_weights.clone()
            weights[hard_pass_idx] = torch.clamp(weights[hard_pass_idx] * pass_boost, min=min_w, max=max_w)
        else:
            weights = base_weights

        return weights

    @torch.no_grad()
    def _collect_val_logits_labels(self):
        """Collect logits and labels over the validation set."""
        self.model.eval()
        logits_list = []
        labels_list = []
        k_neighbors = self.config['model']['structural_stream']['num_neighbors']

        for batch in self.val_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                embeddings, labels, _groups, _idx = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                embeddings, labels, _idx = batch
            else:
                embeddings, labels = batch

            # Move to device (handle multi-field)
            if isinstance(embeddings, list):
                embeddings = [emb.to(self.device) for emb in embeddings]
            else:
                embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            # Build batch k-NN graph
            if isinstance(embeddings, list):
                embeddings_for_graph = self.model.field_fusion(embeddings)
            else:
                embeddings_for_graph = embeddings
            edge_index, edge_weights = self._build_batch_knn_graph(embeddings_for_graph, k=k_neighbors)

            logits = self.model(
                embeddings=embeddings,
                edge_index=edge_index,
                edge_weights=edge_weights
            )
            logits_list.append(logits.detach().cpu())
            labels_list.append(labels.detach().cpu())

        return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)

    def _calibrate_temperature(self, max_iters: int = 200, lr: float = 0.01) -> float:
        """
        Learn a temperature parameter T (>0) on validation logits
        to minimize NLL (cross-entropy). Returns the learned T.
        """
        logits, labels = self._collect_val_logits_labels()
        device = logits.device

        # Optimize log_temperature for positivity of T
        log_temperature = torch.zeros(1, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([log_temperature], lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(max_iters):
            optimizer.zero_grad()
            T = torch.exp(log_temperature)
            loss = criterion(logits / T, labels)
            loss.backward()
            optimizer.step()

        T_final = float(torch.exp(log_temperature).detach().cpu().item())
        # Bound temperature to reasonable range
        T_final = max(0.5, min(5.0, T_final))
        return T_final
