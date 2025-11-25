"""
Composite Metrics Module
Implements holistic evaluation metrics that combine multiple perspectives
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score
)
import logging

logger = logging.getLogger(__name__)


class CompositeMetrics:
    """
    Composite metrics for holistic model evaluation

    Implements combined metrics that ensure balanced assessment:
    - F1_Macro + Recall_Pass: Ensures both overall balance and Pass class detection
    - Harmonic Mean of per-class F1: Penalizes collapse to single class
    - Min-Max Recall Gap: Measures class balance
    """

    def __init__(self, num_classes: int = 2, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute comprehensive set of metrics including composites

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for AUPRC/ROC)

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Standard metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)

        # Per-class metrics
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

        for i, class_name in enumerate(self.class_names):
            metrics[f'f1_{class_name}'] = per_class_f1[i]
            metrics[f'precision_{class_name}'] = per_class_precision[i]
            metrics[f'recall_{class_name}'] = per_class_recall[i]

        # AUPRC/ROC if probabilities provided
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                metrics['roc_auc'] = 0.0

        # COMPOSITE METRICS

        # 1. F1_Macro + Recall_Pass (for binary classification)
        if self.num_classes == 2:
            recall_pass = per_class_recall[1] if len(per_class_recall) > 1 else 0.0
            metrics['composite_f1_recall_pass'] = metrics['f1_macro'] + recall_pass
            metrics['recall_pass'] = recall_pass
            metrics['recall_not_pass'] = per_class_recall[0] if len(per_class_recall) > 0 else 0.0

        # 2. Harmonic Mean of Per-Class F1 (penalizes collapse)
        non_zero_f1 = per_class_f1[per_class_f1 > 0]
        if len(non_zero_f1) > 0:
            metrics['harmonic_mean_f1'] = len(per_class_f1) / np.sum(1.0 / (per_class_f1 + 1e-10))
        else:
            metrics['harmonic_mean_f1'] = 0.0

        # 3. Min-Max Recall Gap (measures class balance)
        metrics['recall_gap'] = per_class_recall.max() - per_class_recall.min()

        # 4. Balanced Accuracy (geometric mean of per-class recall)
        metrics['balanced_accuracy'] = np.sqrt(np.prod(per_class_recall + 1e-10))

        # 5. Prediction Diversity Score (entropy of prediction distribution)
        unique, counts = np.unique(y_pred, return_counts=True)
        proportions = counts / len(y_pred)
        entropy = -np.sum(proportions * np.log(proportions + 1e-10))
        max_entropy = np.log(self.num_classes)
        metrics['prediction_diversity'] = entropy / max_entropy if max_entropy > 0 else 0.0

        # 6. Primary Composite Score (for model selection)
        # Combines F1 Macro (overall performance) + Recall Pass (minority class) + Diversity
        if self.num_classes == 2:
            metrics['primary_composite_score'] = (
                0.5 * metrics['f1_macro'] +
                0.3 * metrics.get('recall_pass', 0.0) +
                0.2 * metrics['prediction_diversity']
            )
        else:
            metrics['primary_composite_score'] = (
                0.6 * metrics['f1_macro'] +
                0.2 * metrics['balanced_accuracy'] +
                0.2 * metrics['prediction_diversity']
            )

        return metrics

    def detect_collapse(self, metrics: Dict) -> Optional[Dict]:
        """
        Detect if model has collapsed based on metrics

        Args:
            metrics: Dictionary of computed metrics

        Returns:
            Collapse information if detected, None otherwise
        """
        collapse_indicators = []

        # Check prediction diversity
        if metrics.get('prediction_diversity', 1.0) < 0.1:
            collapse_indicators.append({
                'indicator': 'LOW_DIVERSITY',
                'value': metrics['prediction_diversity'],
                'threshold': 0.1,
                'severity': 'CRITICAL'
            })

        # Check recall gap (for binary)
        if self.num_classes == 2:
            if metrics.get('recall_gap', 0.0) > 0.8:
                collapse_indicators.append({
                    'indicator': 'HIGH_RECALL_GAP',
                    'value': metrics['recall_gap'],
                    'threshold': 0.8,
                    'severity': 'HIGH'
                })

            # Check if minority class (Pass) has 0 recall
            if metrics.get('recall_pass', 1.0) < 0.05:
                collapse_indicators.append({
                    'indicator': 'ZERO_MINORITY_RECALL',
                    'value': metrics.get('recall_pass', 0.0),
                    'threshold': 0.05,
                    'severity': 'CRITICAL'
                })

        # Check harmonic mean (should be > 0 for all classes predicted)
        if metrics.get('harmonic_mean_f1', 1.0) < 0.1:
            collapse_indicators.append({
                'indicator': 'LOW_HARMONIC_F1',
                'value': metrics['harmonic_mean_f1'],
                'threshold': 0.1,
                'severity': 'HIGH'
            })

        if len(collapse_indicators) > 0:
            return {
                'collapse_detected': True,
                'indicators': collapse_indicators,
                'num_indicators': len(collapse_indicators),
                'max_severity': max(ind['severity'] for ind in collapse_indicators)
            }

        return None

    def get_success_criteria(self, phase: str = "baseline") -> Dict:
        """
        Get success criteria for different experimental phases

        Args:
            phase: Experimental phase (baseline, optimization, excellence)

        Returns:
            Dictionary of criteria thresholds
        """
        criteria = {
            'baseline': {
                'f1_macro': 0.50,
                'accuracy': 0.60,
                'recall_pass': 0.50,
                'recall_gap': 0.30,
                'prediction_diversity': 0.30,
                'primary_composite_score': 0.40
            },
            'optimization': {
                'f1_macro': 0.55,
                'accuracy': 0.70,
                'recall_pass': 0.65,
                'recall_gap': 0.20,
                'prediction_diversity': 0.50,
                'primary_composite_score': 0.50
            },
            'excellence': {
                'f1_macro': 0.60,
                'accuracy': 0.75,
                'recall_pass': 0.70,
                'recall_gap': 0.15,
                'prediction_diversity': 0.60,
                'primary_composite_score': 0.60
            },
            'production': {
                'f1_macro': 0.65,
                'accuracy': 0.75,
                'recall_pass': 0.70,
                'recall_not_pass': 0.70,
                'recall_gap': 0.10,
                'prediction_diversity': 0.70,
                'primary_composite_score': 0.65,
                'harmonic_mean_f1': 0.60
            }
        }

        return criteria.get(phase, criteria['baseline'])

    def evaluate_against_criteria(
        self,
        metrics: Dict,
        phase: str = "baseline"
    ) -> Dict:
        """
        Evaluate metrics against phase-specific success criteria

        Args:
            metrics: Computed metrics
            phase: Experimental phase

        Returns:
            Evaluation results with pass/fail per criterion
        """
        criteria = self.get_success_criteria(phase)
        results = {
            'phase': phase,
            'passed': {},
            'failed': {},
            'overall_pass': True
        }

        for metric_name, threshold in criteria.items():
            if metric_name in metrics:
                metric_value = metrics[metric_name]

                # Special case: recall_gap should be BELOW threshold
                if metric_name == 'recall_gap':
                    passed = metric_value <= threshold
                else:
                    passed = metric_value >= threshold

                if passed:
                    results['passed'][metric_name] = {
                        'value': metric_value,
                        'threshold': threshold,
                        'margin': abs(metric_value - threshold)
                    }
                else:
                    results['failed'][metric_name] = {
                        'value': metric_value,
                        'threshold': threshold,
                        'deficit': abs(threshold - metric_value)
                    }
                    results['overall_pass'] = False

        results['num_passed'] = len(results['passed'])
        results['num_failed'] = len(results['failed'])
        results['pass_rate'] = len(results['passed']) / len(criteria) if len(criteria) > 0 else 0.0

        return results

    def format_metrics_report(self, metrics: Dict, phase: str = "baseline") -> str:
        """
        Generate formatted metrics report

        Args:
            metrics: Computed metrics
            phase: Experimental phase

        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE METRICS REPORT")
        report.append("=" * 80)

        # Standard metrics
        report.append("\nStandard Metrics:")
        report.append(f"  Accuracy:           {metrics.get('accuracy', 0.0):.4f}")
        report.append(f"  F1 Macro:           {metrics.get('f1_macro', 0.0):.4f}")
        report.append(f"  F1 Weighted:        {metrics.get('f1_weighted', 0.0):.4f}")
        report.append(f"  Precision Macro:    {metrics.get('precision_macro', 0.0):.4f}")
        report.append(f"  Recall Macro:       {metrics.get('recall_macro', 0.0):.4f}")

        # Per-class metrics
        report.append("\nPer-Class Metrics:")
        for class_name in self.class_names:
            report.append(f"  {class_name}:")
            report.append(f"    F1:        {metrics.get(f'f1_{class_name}', 0.0):.4f}")
            report.append(f"    Precision: {metrics.get(f'precision_{class_name}', 0.0):.4f}")
            report.append(f"    Recall:    {metrics.get(f'recall_{class_name}', 0.0):.4f}")

        # Composite metrics
        report.append("\nComposite Metrics:")
        report.append(f"  Primary Composite Score:    {metrics.get('primary_composite_score', 0.0):.4f}")
        report.append(f"  Harmonic Mean F1:           {metrics.get('harmonic_mean_f1', 0.0):.4f}")
        report.append(f"  Balanced Accuracy:          {metrics.get('balanced_accuracy', 0.0):.4f}")
        report.append(f"  Recall Gap:                 {metrics.get('recall_gap', 0.0):.4f}")
        report.append(f"  Prediction Diversity:       {metrics.get('prediction_diversity', 0.0):.4f}")

        # Evaluation against criteria
        evaluation = self.evaluate_against_criteria(metrics, phase)
        report.append("\n" + "=" * 80)
        report.append(f"EVALUATION AGAINST {phase.upper()} CRITERIA")
        report.append("=" * 80)
        report.append(f"Pass Rate: {evaluation['pass_rate']:.1%} ({evaluation['num_passed']}/{len(evaluation['passed']) + len(evaluation['failed'])})")
        report.append(f"Overall: {'✓ PASS' if evaluation['overall_pass'] else '✗ FAIL'}")

        if evaluation['passed']:
            report.append("\n✓ Passed Criteria:")
            for metric, info in evaluation['passed'].items():
                report.append(f"  {metric:30s}: {info['value']:.4f} (threshold: {info['threshold']:.4f})")

        if evaluation['failed']:
            report.append("\n✗ Failed Criteria:")
            for metric, info in evaluation['failed'].items():
                report.append(f"  {metric:30s}: {info['value']:.4f} (threshold: {info['threshold']:.4f}, deficit: {info['deficit']:.4f})")

        # Collapse detection
        collapse = self.detect_collapse(metrics)
        if collapse:
            report.append("\n" + "!" * 80)
            report.append("⚠️  COLLAPSE DETECTED!")
            report.append("!" * 80)
            for indicator in collapse['indicators']:
                report.append(f"  {indicator['indicator']:25s}: {indicator['value']:.4f} (severity: {indicator['severity']})")
            report.append("!" * 80)

        report.append("=" * 80)

        return "\n".join(report)
