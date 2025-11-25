"""
SMOTE Handler Module
Applies resampling methods (SMOTE, ADASYN, BorderlineSMOTE) to embeddings for class balancing
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    RESAMPLING_AVAILABLE = True
except ImportError:
    RESAMPLING_AVAILABLE = False
    logger.warning("imbalanced-learn not installed. Resampling methods will not be available.")


def apply_smote_to_embeddings(embeddings: np.ndarray, labels: np.ndarray, config: dict) -> tuple:
    """
    Apply resampling method (SMOTE, ADASYN, BorderlineSMOTE) to embeddings to balance classes

    Args:
        embeddings: Numpy array of embeddings (N, D)
        labels: Numpy array of labels (N,)
        config: Configuration dictionary

    Returns:
        Tuple of (balanced_embeddings, balanced_labels)
    """
    if not RESAMPLING_AVAILABLE:
        logger.warning("Resampling methods not available. Returning original data.")
        return embeddings, labels

    # Check if resampling is enabled
    use_resampling = config['data'].get('use_smote', False) or config['data'].get('use_resampling', False)
    if not use_resampling:
        logger.info("Resampling disabled in config. Returning original data.")
        return embeddings, labels

    logger.info(f"Original data shape: {embeddings.shape}")
    logger.info(f"Original class distribution: {np.bincount(labels)}")

    # Get resampling parameters from config
    k_neighbors = config['data'].get('smote_k_neighbors', 5)
    sampling_strategy = config['data'].get('smote_sampling_strategy', 'auto')
    resampling_method = config['data'].get('resampling_method', 'smote').lower()
    random_seed = config['data'].get('random_seed', 42)

    try:
        # Initialize the appropriate resampling method
        if resampling_method == 'adasyn':
            logger.info("Using ADASYN for resampling")
            resampler = ADASYN(
                sampling_strategy=sampling_strategy,
                n_neighbors=k_neighbors,
                random_state=random_seed
            )
        elif resampling_method == 'borderline_smote' or resampling_method == 'borderline':
            logger.info("Using BorderlineSMOTE for resampling")
            resampler = BorderlineSMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=random_seed
            )
        elif resampling_method == 'smote':
            logger.info("Using SMOTE for resampling")
            resampler = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=random_seed
            )
        else:
            logger.warning(f"Unknown resampling method: {resampling_method}. Using SMOTE.")
            resampler = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=random_seed
            )

        # Apply resampling
        embeddings_balanced, labels_balanced = resampler.fit_resample(embeddings, labels)

        logger.info(f"Balanced data shape: {embeddings_balanced.shape}")
        logger.info(f"Balanced class distribution: {np.bincount(labels_balanced)}")

        return embeddings_balanced, labels_balanced

    except Exception as e:
        logger.error(f"Error applying {resampling_method}: {e}")
        logger.warning("Returning original data without resampling.")
        return embeddings, labels
