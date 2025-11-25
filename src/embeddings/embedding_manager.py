"""
Embedding Manager with Automatic Caching

High-level interface for embedding generation with:
- Automatic cache detection and reuse
- Intelligent regeneration when data changes
- Force regeneration option
- Progress tracking and logging
"""

import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from .embedding_cache import EmbeddingCache
from .sbert_encoder import SBERTEncoder

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embedding generation with intelligent caching

    Usage:
        manager = EmbeddingManager(config, force_regenerate=False)
        embeddings = manager.get_embeddings(train_df, test_df)
    """

    def __init__(self, config: Dict, force_regenerate: bool = False, cache_dir: str = 'cache'):
        """
        Initialize embedding manager

        Args:
            config: Configuration dictionary
            force_regenerate: If True, regenerate embeddings even if cached
            cache_dir: Directory for cache storage
        """
        self.config = config
        self.force_regenerate = force_regenerate
        self.cache = EmbeddingCache(cache_dir=cache_dir) if cache_dir is not None else None

        # Get embedding config
        self.embedding_config = config.get('embedding', config.get('semantic', {}))
        self.model_name = self.embedding_config.get('model_name', 'sentence-transformers/all-mpnet-base-v2')
        self.device = self.embedding_config.get('device', 'cuda')

    def _prepare_tc_texts(self, df: pd.DataFrame) -> list:
        """Prepare test case texts from dataframe"""
        texts = []
        for _, row in df.iterrows():
            summary = row.get('tc_summary', row.get('summary', ''))
            steps = row.get('tc_steps', row.get('steps', ''))

            if summary and steps:
                text = f"Summary: {summary}\nSteps: {steps}"
            elif summary:
                text = f"Summary: {summary}"
            elif steps:
                text = f"Steps: {steps}"
            else:
                text = "No test case information"

            texts.append(text)

        return texts

    def _prepare_commit_texts(self, df: pd.DataFrame) -> list:
        """Prepare commit texts from dataframe"""
        texts = []
        for _, row in df.iterrows():
            msg = row.get('commit_msg', row.get('message', ''))
            diff = row.get('commit_diff', row.get('diff', ''))

            if msg and diff:
                # Truncate diff to 2000 chars (SBERT max is 512 tokens, ~2000 chars)
                diff_truncated = diff[:2000] if len(diff) > 2000 else diff
                text = f"Commit Message: {msg}\n\nDiff:\n{diff_truncated}"
            elif msg:
                text = f"Commit Message: {msg}"
            elif diff:
                diff_truncated = diff[:2000] if len(diff) > 2000 else diff
                text = f"Diff:\n{diff_truncated}"
            else:
                text = "No commit information"

            texts.append(text)

        return texts

    def _generate_embeddings(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple:
        """
        Generate embeddings from scratch

        Args:
            train_df: Training dataframe
            test_df: Test dataframe

        Returns:
            tuple: (train_tc_emb, test_tc_emb, train_commit_emb, test_commit_emb, embedding_dim, model_name)
        """
        logger.info("="*70)
        logger.info("GENERATING EMBEDDINGS")
        logger.info("="*70)

        # Initialize encoder
        logger.info(f"Initializing encoder: {self.model_name}")
        encoder = SBERTEncoder(self.config, device=self.device)

        # Prepare texts
        logger.info("Preparing texts...")
        train_tc_texts = self._prepare_tc_texts(train_df)
        test_tc_texts = self._prepare_tc_texts(test_df)
        train_commit_texts = self._prepare_commit_texts(train_df)
        test_commit_texts = self._prepare_commit_texts(test_df)

        logger.info(f"  Train TCs: {len(train_tc_texts)}")
        logger.info(f"  Test TCs: {len(test_tc_texts)}")
        logger.info(f"  Train Commits: {len(train_commit_texts)}")
        logger.info(f"  Test Commits: {len(test_commit_texts)}")

        # Get chunk size from config
        batch_size = self.embedding_config.get('batch_size', 128)
        chunk_size = batch_size * 10  # 10 batches per chunk

        # Encode
        logger.info("Encoding...")
        train_tc_emb = encoder.encode_texts_chunked(train_tc_texts, chunk_size=chunk_size, desc="Train TCs")
        test_tc_emb = encoder.encode_texts_chunked(test_tc_texts, chunk_size=chunk_size, desc="Test TCs")
        train_commit_emb = encoder.encode_texts_chunked(train_commit_texts, chunk_size=chunk_size, desc="Train Commits")
        test_commit_emb = encoder.encode_texts_chunked(test_commit_texts, chunk_size=chunk_size, desc="Test Commits")

        embedding_dim = encoder.get_embedding_dim()

        logger.info("="*70)

        return train_tc_emb, test_tc_emb, train_commit_emb, test_commit_emb, embedding_dim, self.model_name

    def get_embeddings(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get embeddings (from cache or generate new)

        Automatically:
        1. Checks if cache exists and is valid
        2. Reuses cache if available and valid
        3. Regenerates if cache invalid or force_regenerate=True
        4. Saves newly generated embeddings to cache

        Args:
            train_df: Training dataframe
            test_df: Test dataframe

        Returns:
            embeddings: Dictionary with keys:
                - 'train_tc': Train TC embeddings
                - 'test_tc': Test TC embeddings
                - 'train_commit': Train commit embeddings
                - 'test_commit': Test commit embeddings
                - 'embedding_dim': Embedding dimension
                - 'model_name': Model name
        """
        # Check cache
        use_cache = False

        if self.cache is None:
            logger.info("Cache disabled - generating embeddings")
        elif self.force_regenerate:
            logger.info("Force regenerate enabled - ignoring cache")
        elif self.cache.exists():
            if self.cache.is_valid(train_df, test_df):
                logger.info("Valid cache found - loading embeddings from cache")
                use_cache = True
            else:
                logger.info("Cache found but invalid (data changed) - regenerating")
        else:
            logger.info("No cache found - generating embeddings")

        # Load or generate
        if use_cache:
            train_tc_emb, test_tc_emb, train_commit_emb, test_commit_emb, embedding_dim, model_name = self.cache.load()
        else:
            train_tc_emb, test_tc_emb, train_commit_emb, test_commit_emb, embedding_dim, model_name = self._generate_embeddings(train_df, test_df)

            # Save to cache (if enabled)
            if self.cache is not None:
                self.cache.save(
                    train_tc_emb, test_tc_emb, train_commit_emb, test_commit_emb,
                    embedding_dim, model_name, train_df, test_df
                )

        # Return as dictionary
        return {
            'train_tc': train_tc_emb,
            'test_tc': test_tc_emb,
            'train_commit': train_commit_emb,
            'test_commit': test_commit_emb,
            'embedding_dim': embedding_dim,
            'model_name': model_name
        }

    def clear_cache(self):
        """Clear embedding cache"""
        if self.cache is not None:
            self.cache.clear()

    def cache_info(self) -> str:
        """Get cache information"""
        if self.cache is not None:
            return self.cache.info()
        else:
            return "Cache disabled"
