#!/usr/bin/env python3
"""
Extract metrics from all experiments (007-011)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
import json
from pathlib import Path

def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate all metrics"""
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))

    # Per-class metrics
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()

        # Class 0 (Not-Pass)
        metrics['not_pass_precision'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        metrics['not_pass_recall'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics['not_pass_f1'] = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

        # Class 1 (Pass)
        metrics['pass_precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        metrics['pass_recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics['pass_f1'] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

        # Support
        metrics['not_pass_support'] = int(tn + fp)
        metrics['pass_support'] = int(tp + fn)
        metrics['total_support'] = int(len(y_true))

        # Confusion matrix
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

    # AUC metrics
    try:
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc_macro'] = float(roc_auc_score(y_true, y_proba, average='macro'))
            metrics['roc_auc_weighted'] = float(roc_auc_score(y_true, y_proba, average='weighted'))
            metrics['auprc_macro'] = float(average_precision_score(y_true, y_proba, average='macro'))
            metrics['auprc_weighted'] = float(average_precision_score(y_true, y_proba, average='weighted'))
    except:
        pass

    # Prediction diversity
    class_dist = np.bincount(y_pred) / len(y_pred)
    metrics['prediction_diversity'] = float(1 - np.max(class_dist))

    return metrics


def extract_experiment_metrics(exp_dir):
    """Extract metrics from a single experiment"""
    exp_path = Path(exp_dir)
    predictions_file = exp_path / "predictions.npz"

    if not predictions_file.exists():
        return None

    # Load predictions
    data = np.load(predictions_file)

    # Extract test set predictions
    test_true = data['test_labels']
    test_pred = data['test_predictions']
    test_proba = data['test_probabilities'][:, 1]  # Probability of class 1

    # Calculate metrics
    metrics = calculate_metrics(test_true, test_pred, test_proba)

    return metrics


def main():
    experiments = [
        "experiment_007_phase1_stabilization",
        "experiment_008_gatv2",
        "experiment_009_attention_pooling",
        "experiment_010_bidirectional_fusion",
        "experiment_011_improved_classifier"
    ]

    results = {}

    print("="*80)
    print("EXTRACTING METRICS FROM ALL EXPERIMENTS (007-011)")
    print("="*80)
    print()

    for exp_name in experiments:
        exp_path = Path("results") / exp_name

        print(f"Processing: {exp_name}")

        metrics = extract_experiment_metrics(exp_path)

        if metrics:
            results[exp_name] = metrics

            # Display key metrics
            print(f"  ✓ Metrics extracted")
            print(f"    - Test Accuracy:   {metrics['accuracy']:.4f}")
            print(f"    - Test F1 Macro:   {metrics['f1_macro']:.4f}")
            print(f"    - Test AUPRC:      {metrics.get('auprc_macro', 0.0):.4f}")
            print(f"    - Pass Recall:     {metrics.get('pass_recall', 0.0):.4f}")
            print(f"    - Not-Pass Recall: {metrics.get('not_pass_recall', 0.0):.4f}")
        else:
            print(f"  ✗ No predictions found")

        print()

    # Save all results
    output_file = Path("reports/all_experiments_metrics.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("="*80)
    print(f"✓ All metrics saved to: {output_file}")
    print("="*80)

    # Summary comparison
    print()
    print("="*80)
    print("QUICK COMPARISON (F1 MACRO)")
    print("="*80)
    print()

    if results:
        baseline_f1 = results.get("experiment_007_phase1_stabilization", {}).get("f1_macro", 0.0)

        print(f"{'Experiment':<40} {'F1 Macro':<12} {'Delta':<12} {'Status'}")
        print("-"*80)

        for exp_name in experiments:
            if exp_name in results:
                f1 = results[exp_name]['f1_macro']
                delta = f1 - baseline_f1

                if exp_name == "experiment_007_phase1_stabilization":
                    status = "BASELINE"
                elif delta > 0:
                    status = f"+{delta:.4f} ✓"
                elif delta < 0:
                    status = f"{delta:.4f} ✗"
                else:
                    status = "NO CHANGE"

                print(f"{exp_name:<40} {f1:.4f}       {status}")

    print()


if __name__ == "__main__":
    main()
