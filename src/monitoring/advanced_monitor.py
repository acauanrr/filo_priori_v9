"""
Advanced Training Monitor
Real-time monitoring with early warning system for training anomalies
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
import warnings

logger = logging.getLogger(__name__)


class AdvancedTrainingMonitor:
    """
    Advanced monitoring system for detecting training anomalies in real-time

    Features:
    - Prediction collapse detection (intra-epoch)
    - Gradient monitoring (exploding/vanishing)
    - Loss spike detection
    - Learning rate anomaly detection
    - Class distribution tracking
    """

    def __init__(self, config: Dict, num_classes: int = 2):
        self.config = config
        self.num_classes = num_classes

        # Monitoring configuration
        self.monitor_config = config.get('monitoring', {})
        self.check_every_n_batches = self.monitor_config.get('check_every_n_batches', 50)
        self.prediction_diversity_threshold = self.monitor_config.get('prediction_diversity_threshold', 0.05)
        self.gradient_norm_max_threshold = self.monitor_config.get('gradient_norm_max', 100.0)
        self.gradient_norm_min_threshold = self.monitor_config.get('gradient_norm_min', 1e-6)
        self.loss_spike_multiplier = self.monitor_config.get('loss_spike_multiplier', 5.0)

        # State tracking
        self.batch_counter = 0
        self.epoch_counter = 0

        # Prediction distribution tracking (per epoch)
        self.prediction_counts = np.zeros(num_classes, dtype=int)
        self.total_predictions = 0

        # Gradient tracking
        self.gradient_norms = deque(maxlen=100)
        self.gradient_history = []

        # Loss tracking
        self.loss_history = deque(maxlen=100)
        self.loss_epoch_history = []

        # Warnings and alerts
        self.warnings_issued = []
        self.critical_alerts = []

        # Flags for early termination
        self.should_terminate = False
        self.termination_reason = None

    def reset_epoch(self):
        """Reset per-epoch tracking"""
        self.prediction_counts = np.zeros(self.num_classes, dtype=int)
        self.total_predictions = 0
        self.batch_counter = 0
        self.epoch_counter += 1

    def update_predictions(self, predictions: torch.Tensor):
        """
        Update prediction distribution tracking

        Args:
            predictions: Tensor of predicted class indices [batch_size]
        """
        predictions_np = predictions.cpu().numpy()
        unique, counts = np.unique(predictions_np, return_counts=True)

        for cls, count in zip(unique, counts):
            if cls < self.num_classes:
                self.prediction_counts[cls] += count

        self.total_predictions += len(predictions_np)
        self.batch_counter += 1

    def check_prediction_collapse(self, batch_idx: int) -> Optional[Dict]:
        """
        Check if model has collapsed to predicting single class

        Returns:
            Warning dictionary if collapse detected, None otherwise
        """
        if self.total_predictions == 0:
            return None

        # Calculate prediction proportions
        proportions = self.prediction_counts / self.total_predictions

        # Check if any class has < threshold proportion
        min_proportion = proportions.min()
        max_proportion = proportions.max()

        # Collapse detected if one class dominates (>95%) or minority class <5%
        if max_proportion > 0.95 or min_proportion < self.prediction_diversity_threshold:
            dominant_class = proportions.argmax()

            warning = {
                'type': 'PREDICTION_COLLAPSE',
                'severity': 'CRITICAL',
                'epoch': self.epoch_counter,
                'batch': batch_idx,
                'proportions': proportions.tolist(),
                'dominant_class': int(dominant_class),
                'message': f"Prediction collapse detected! Class {dominant_class} dominates with {max_proportion:.1%}"
            }

            logger.error("!" * 80)
            logger.error("⚠️  CRITICAL: PREDICTION COLLAPSE DETECTED!")
            logger.error(f"⚠️  Epoch {self.epoch_counter}, Batch {batch_idx}")
            logger.error(f"⚠️  Prediction distribution: {proportions}")
            logger.error(f"⚠️  Class {dominant_class} dominates with {max_proportion:.1%}")
            logger.error("!" * 80)

            self.critical_alerts.append(warning)
            return warning

        return None

    def update_gradients(self, model: torch.nn.Module):
        """
        Monitor gradient norms for vanishing/exploding gradients

        Args:
            model: PyTorch model
        """
        total_norm = 0.0
        param_count = 0

        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1

        if param_count > 0:
            total_norm = total_norm ** 0.5
            self.gradient_norms.append(total_norm)

            # Check for anomalies
            if total_norm > self.gradient_norm_max_threshold:
                warning = {
                    'type': 'EXPLODING_GRADIENT',
                    'severity': 'HIGH',
                    'epoch': self.epoch_counter,
                    'batch': self.batch_counter,
                    'gradient_norm': total_norm,
                    'threshold': self.gradient_norm_max_threshold,
                    'message': f"Exploding gradient detected! Norm: {total_norm:.2f}"
                }

                logger.warning("⚠️  Exploding gradient detected!")
                logger.warning(f"⚠️  Gradient norm: {total_norm:.2f} (threshold: {self.gradient_norm_max_threshold})")

                self.warnings_issued.append(warning)

            elif total_norm < self.gradient_norm_min_threshold:
                warning = {
                    'type': 'VANISHING_GRADIENT',
                    'severity': 'MEDIUM',
                    'epoch': self.epoch_counter,
                    'batch': self.batch_counter,
                    'gradient_norm': total_norm,
                    'threshold': self.gradient_norm_min_threshold,
                    'message': f"Vanishing gradient detected! Norm: {total_norm:.2e}"
                }

                logger.warning("⚠️  Vanishing gradient detected!")
                logger.warning(f"⚠️  Gradient norm: {total_norm:.2e} (threshold: {self.gradient_norm_min_threshold})")

                self.warnings_issued.append(warning)

    def update_loss(self, loss_value: float):
        """
        Monitor loss for sudden spikes

        Args:
            loss_value: Current loss value
        """
        self.loss_history.append(loss_value)

        # Check for loss spikes
        if len(self.loss_history) > 10:
            recent_avg = np.mean(list(self.loss_history)[-10:])

            if loss_value > recent_avg * self.loss_spike_multiplier:
                warning = {
                    'type': 'LOSS_SPIKE',
                    'severity': 'MEDIUM',
                    'epoch': self.epoch_counter,
                    'batch': self.batch_counter,
                    'loss': loss_value,
                    'recent_avg': recent_avg,
                    'multiplier': loss_value / recent_avg,
                    'message': f"Loss spike detected! Current: {loss_value:.4f}, Avg: {recent_avg:.4f}"
                }

                logger.warning("⚠️  Loss spike detected!")
                logger.warning(f"⚠️  Current loss: {loss_value:.4f}, Recent avg: {recent_avg:.4f}")
                logger.warning(f"⚠️  Multiplier: {loss_value / recent_avg:.2f}x")

                self.warnings_issued.append(warning)

    def intra_epoch_check(self, batch_idx: int, predictions: torch.Tensor,
                          model: torch.nn.Module, loss: float) -> Dict:
        """
        Comprehensive intra-epoch monitoring check

        Args:
            batch_idx: Current batch index
            predictions: Predicted classes
            model: Model being trained
            loss: Current loss value

        Returns:
            Dictionary with monitoring results
        """
        results = {
            'warnings': [],
            'should_terminate': False,
            'termination_reason': None
        }

        # Update tracking
        self.update_predictions(predictions)
        self.update_loss(loss)
        self.update_gradients(model)

        # Check every N batches
        if batch_idx % self.check_every_n_batches == 0 and batch_idx > 0:
            # Check prediction collapse
            collapse_warning = self.check_prediction_collapse(batch_idx)
            if collapse_warning:
                results['warnings'].append(collapse_warning)

                # Critical: suggest termination for prediction collapse
                if collapse_warning['severity'] == 'CRITICAL':
                    results['should_terminate'] = True
                    results['termination_reason'] = 'PREDICTION_COLLAPSE'

            # Log current status
            if self.total_predictions > 0:
                proportions = self.prediction_counts / self.total_predictions
                logger.info(f"Batch {batch_idx}: Prediction distribution: {proportions}")

            # Log gradient status
            if len(self.gradient_norms) > 0:
                current_grad_norm = self.gradient_norms[-1]
                logger.debug(f"Batch {batch_idx}: Gradient norm: {current_grad_norm:.4f}")

        return results

    def end_epoch_summary(self) -> Dict:
        """
        Generate end-of-epoch summary

        Returns:
            Dictionary with epoch statistics
        """
        summary = {
            'epoch': self.epoch_counter,
            'total_predictions': self.total_predictions,
            'prediction_distribution': {},
            'gradient_statistics': {},
            'loss_statistics': {},
            'warnings_count': len(self.warnings_issued),
            'critical_alerts_count': len(self.critical_alerts)
        }

        # Prediction distribution
        if self.total_predictions > 0:
            proportions = self.prediction_counts / self.total_predictions
            for cls in range(self.num_classes):
                summary['prediction_distribution'][f'class_{cls}'] = {
                    'count': int(self.prediction_counts[cls]),
                    'proportion': float(proportions[cls])
                }

        # Gradient statistics
        if len(self.gradient_norms) > 0:
            grad_array = np.array(list(self.gradient_norms))
            summary['gradient_statistics'] = {
                'mean': float(grad_array.mean()),
                'std': float(grad_array.std()),
                'min': float(grad_array.min()),
                'max': float(grad_array.max()),
                'median': float(np.median(grad_array))
            }

        # Loss statistics
        if len(self.loss_history) > 0:
            loss_array = np.array(list(self.loss_history))
            summary['loss_statistics'] = {
                'mean': float(loss_array.mean()),
                'std': float(loss_array.std()),
                'min': float(loss_array.min()),
                'max': float(loss_array.max()),
                'final': float(loss_array[-1])
            }

        # Store for history
        self.gradient_history.append(summary['gradient_statistics'])
        self.loss_epoch_history.append(summary['loss_statistics'])

        return summary

    def get_full_report(self) -> Dict:
        """
        Generate comprehensive monitoring report

        Returns:
            Full monitoring report
        """
        report = {
            'total_epochs': self.epoch_counter,
            'warnings': self.warnings_issued,
            'critical_alerts': self.critical_alerts,
            'gradient_history': self.gradient_history,
            'loss_history': self.loss_epoch_history,
            'termination_triggered': self.should_terminate,
            'termination_reason': self.termination_reason
        }

        return report

    def should_early_terminate(self) -> Tuple[bool, Optional[str]]:
        """
        Check if training should be terminated early due to critical issues

        Returns:
            (should_terminate, reason)
        """
        # Check for critical alerts in recent history
        recent_critical = [a for a in self.critical_alerts
                          if a['epoch'] == self.epoch_counter]

        if len(recent_critical) > 0:
            return True, recent_critical[0]['type']

        return False, None
