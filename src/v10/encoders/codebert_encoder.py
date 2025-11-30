"""
CodeBERT Encoder for Filo-Priori V10.

This module implements a CodeBERT-based encoder for code semantics,
replacing SBERT from V9.

Key Differences from SBERT (V9):
    - Pre-trained on code (CodeSearchNet), not natural language
    - Better understanding of programming syntax and semantics
    - Handles code-specific patterns (CamelCase, snake_case)

CodeBERT Reference:
    Feng et al., "CodeBERT: A Pre-Trained Model for Programming
    and Natural Languages", EMNLP 2020.

Model Options:
    - microsoft/codebert-base (768-dim, general purpose)
    - microsoft/graphcodebert-base (structure-aware)
    - microsoft/unixcoder-base (unified code understanding)
"""

import logging
from typing import List, Optional, Dict, Union, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


class CodeBERTEncoder(nn.Module):
    """
    CodeBERT-based encoder for code and test semantics.

    This encoder processes:
    1. Test identifiers (class names, method names)
    2. Code changes (diffs, file names)
    3. Combined test-code pairs (for co-attention)

    Args:
        model_name: HuggingFace model name.
        output_dim: Output embedding dimension.
        pooling: Pooling strategy ('cls', 'mean', 'max').
        freeze_layers: Number of layers to freeze (0 = none).
        use_cache: Whether to cache embeddings.
        cache_dir: Directory for model/embedding cache.

    Example:
        >>> encoder = CodeBERTEncoder()
        >>> embeddings = encoder.encode(["getUserById", "createOrder"])
        >>> print(embeddings.shape)  # [2, 768]
    """

    SUPPORTED_MODELS = {
        'codebert': 'microsoft/codebert-base',
        'graphcodebert': 'microsoft/graphcodebert-base',
        'unixcoder': 'microsoft/unixcoder-base',
        'codet5': 'Salesforce/codet5-base',
    }

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        output_dim: int = 768,
        pooling: str = 'cls',
        freeze_layers: int = 0,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        max_length: int = 256,
        device: Optional[str] = None
    ):
        super().__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.pooling = pooling
        self.freeze_layers = freeze_layers
        self.use_cache = use_cache
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Lazy loading of transformers
        self._tokenizer = None
        self._encoder = None
        self._is_loaded = False

        # Embedding cache
        self._cache: Dict[str, torch.Tensor] = {}

        # Optional projection layer
        self.projection = None

        logger.info(f"Initialized CodeBERTEncoder with model: {model_name}")

    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._is_loaded:
            return

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "transformers package required. Install with: pip install transformers"
            )

        logger.info(f"Loading CodeBERT model: {self.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._encoder = AutoModel.from_pretrained(self.model_name)
        self._encoder.to(self.device)

        # Freeze layers if requested
        if self.freeze_layers > 0:
            self._freeze_encoder_layers(self.freeze_layers)

        # Add projection if output_dim differs
        encoder_dim = self._encoder.config.hidden_size
        if self.output_dim != encoder_dim:
            self.projection = nn.Linear(encoder_dim, self.output_dim).to(self.device)

        self._is_loaded = True
        logger.info(f"Model loaded. Encoder dim: {encoder_dim}, Output dim: {self.output_dim}")

    def _freeze_encoder_layers(self, num_layers: int):
        """Freeze the first N transformer layers."""
        for name, param in self._encoder.named_parameters():
            if 'embeddings' in name:
                param.requires_grad = False
            elif 'encoder.layer' in name:
                layer_num = int(name.split('.')[2])
                if layer_num < num_layers:
                    param.requires_grad = False

        logger.info(f"Froze first {num_layers} encoder layers")

    def forward(
        self,
        input_texts: List[str],
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the encoder.

        Args:
            input_texts: List of text strings to encode.
            return_attention: Whether to return attention weights.

        Returns:
            Tensor of embeddings [batch_size, output_dim].
            Optionally also returns attention weights.
        """
        return self.encode(input_texts, return_attention=return_attention)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_attention: bool = False,
        show_progress: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings.
            batch_size: Batch size for encoding.
            return_attention: Whether to return attention weights.
            show_progress: Show progress bar.

        Returns:
            Embeddings tensor [num_texts, output_dim].
        """
        self._load_model()

        if not texts:
            return torch.zeros((0, self.output_dim), device=self.device)

        # Check cache
        if self.use_cache:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                if text in self._cache:
                    cached_embeddings.append((i, self._cache[text]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            if not uncached_texts:
                # All cached
                embeddings = torch.zeros((len(texts), self.output_dim), device=self.device)
                for i, emb in cached_embeddings:
                    embeddings[i] = emb
                return embeddings

            texts_to_encode = uncached_texts
        else:
            texts_to_encode = texts
            uncached_indices = list(range(len(texts)))

        # Encode in batches
        all_embeddings = []
        all_attentions = [] if return_attention else None

        iterator = range(0, len(texts_to_encode), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Encoding")
            except ImportError:
                pass

        for i in iterator:
            batch_texts = texts_to_encode[i:i + batch_size]

            with torch.no_grad():
                embeddings, attentions = self._encode_batch(
                    batch_texts, return_attention=return_attention
                )

            all_embeddings.append(embeddings)
            if return_attention and attentions is not None:
                all_attentions.append(attentions)

        # Concatenate batches
        encoded = torch.cat(all_embeddings, dim=0)

        # Update cache
        if self.use_cache:
            for i, text in enumerate(texts_to_encode):
                self._cache[text] = encoded[i].clone()

        # Reconstruct full result (cached + newly encoded)
        if self.use_cache and cached_embeddings:
            full_embeddings = torch.zeros((len(texts), self.output_dim), device=self.device)

            # Add cached
            for i, emb in cached_embeddings:
                full_embeddings[i] = emb

            # Add newly encoded
            for j, idx in enumerate(uncached_indices):
                full_embeddings[idx] = encoded[j]

            result = full_embeddings
        else:
            result = encoded

        if return_attention and all_attentions:
            return result, torch.cat(all_attentions, dim=0)

        return result

    def _encode_batch(
        self,
        texts: List[str],
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode a single batch of texts."""
        # Tokenize
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        # Forward pass
        with autocast(enabled=torch.cuda.is_available()):
            outputs = self._encoder(
                **inputs,
                output_attentions=return_attention
            )

        # Pool embeddings
        if self.pooling == 'cls':
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        elif self.pooling == 'mean':
            # Mean pooling over non-padding tokens
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1)
            embeddings = embeddings / attention_mask.sum(dim=1)
        elif self.pooling == 'max':
            embeddings = outputs.last_hidden_state.max(dim=1)[0]
        else:
            embeddings = outputs.last_hidden_state[:, 0, :]

        # Project if needed
        if self.projection is not None:
            embeddings = self.projection(embeddings)

        # Get attention weights if requested
        attentions = None
        if return_attention and hasattr(outputs, 'attentions') and outputs.attentions:
            # Average attention across heads and layers
            attentions = torch.stack(outputs.attentions).mean(dim=[0, 2])

        return embeddings, attentions

    def encode_pair(
        self,
        texts_a: List[str],
        texts_b: List[str],
        separator: str = " [SEP] "
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode pairs of texts (e.g., test + code).

        Returns separate embeddings for each text in the pair.

        Args:
            texts_a: First texts (e.g., test names).
            texts_b: Second texts (e.g., code changes).
            separator: Separator token between texts.

        Returns:
            Tuple of (embeddings_a, embeddings_b).
        """
        self._load_model()

        # Encode individually first
        embeddings_a = self.encode(texts_a)
        embeddings_b = self.encode(texts_b)

        return embeddings_a, embeddings_b

    def get_sequence_embeddings(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        """
        Get full sequence embeddings (not just [CLS]).

        Useful for co-attention which needs token-level embeddings.

        Returns:
            Tensor [batch, max_length, hidden_dim]
        """
        self._load_model()

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._encoder(**inputs)

        return outputs.last_hidden_state

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    def save_cache(self, path: str):
        """Save cache to disk."""
        torch.save(self._cache, path)
        logger.info(f"Saved cache to {path} ({len(self._cache)} entries)")

    def load_cache(self, path: str):
        """Load cache from disk."""
        if Path(path).exists():
            self._cache = torch.load(path)
            logger.info(f"Loaded cache from {path} ({len(self._cache)} entries)")

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.output_dim


class GraphCodeBERTEncoder(CodeBERTEncoder):
    """
    GraphCodeBERT variant that incorporates data flow information.

    This model was pre-trained with data flow graphs, making it
    better at understanding code structure and dependencies.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('model_name', 'microsoft/graphcodebert-base')
        super().__init__(**kwargs)


class UnixCoderEncoder(CodeBERTEncoder):
    """
    UniXcoder variant with unified cross-modal pre-training.

    Supports multiple programming languages and modalities.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('model_name', 'microsoft/unixcoder-base')
        super().__init__(**kwargs)
