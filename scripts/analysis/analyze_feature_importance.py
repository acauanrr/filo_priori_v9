#!/usr/bin/env python3
"""
Feature Importance Analysis for Experiment 05 (29 Features)

Analyzes which of the 29 structural features are most important for
test case prioritization and recommends a top-10 subset.

Methods:
1. Weight Magnitude Analysis: Analyze first layer weights of structural stream
2. Feature Variance Analysis: Features with low variance are less informative
3. Feature Correlation: Correlation with target (failure prediction)
4. Combined Ranking: Weighted combination of all methods

Author: Filo-Priori V8 Team
Date: 2025-11-14
"""

import torch
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Feature names for V2 (29 features)
FEATURE_NAMES = [
    # PHYLOGENETIC (20)
    'test_age',                    # 0
    'failure_rate',                # 1
    'recent_failure_rate',         # 2
    'flakiness_rate',              # 3
    'execution_count',             # 4
    'failure_count',               # 5
    'pass_count',                  # 6
    'consecutive_failures',        # 7
    'consecutive_passes',          # 8
    'max_consecutive_failures',    # 9
    'last_failure_age',            # 10
    'last_pass_age',               # 11
    'execution_frequency',         # 12
    'failure_trend',               # 13
    'recent_execution_count',      # 14
    'very_recent_failure_rate',    # 15
    'medium_term_failure_rate',    # 16
    'acceleration',                # 17
    'deceleration_factor',         # 18
    'builds_since_change',         # 19
    # STRUCTURAL (9)
    'commit_count',                # 20
    'test_novelty',                # 21
    'builds_affected',             # 22
    'cr_count',                    # 23
    'commit_count_actual',         # 24
    'avg_commits_per_execution',   # 25
    'recent_commit_surge',         # 26
    'stability_score',             # 27
    'pass_fail_ratio',             # 28
]


class FeatureImportanceAnalyzer:
    """Analyze feature importance from trained model and data statistics."""

    def __init__(self, model_path: str, features_cache_path: str):
        """
        Initialize analyzer.

        Args:
            model_path: Path to trained model checkpoint (best_model.pt)
            features_cache_path: Path to cached structural features (V2)
        """
        self.model_path = model_path
        self.features_cache_path = features_cache_path
        self.model = None
        self.structural_weights = None
        self.feature_stats = None

    def load_model(self):
        """Load trained model and extract structural stream weights."""
        print(f"Loading model from: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location='cpu')

        # Extract structural stream first layer weights
        # Key: 'structural_stream.layers.0.weight' → shape [hidden_dim, 29]
        key = 'structural_stream.layers.0.weight'

        if key in checkpoint:
            self.structural_weights = checkpoint[key].numpy()  # [hidden_dim, 29]
            print(f"✓ Loaded structural weights: {self.structural_weights.shape}")
        else:
            print(f"⚠️  Key '{key}' not found. Available keys:")
            print([k for k in checkpoint.keys() if 'structural' in k])

            # Try alternative key
            alt_key = 'model.structural_stream.layers.0.weight'
            if alt_key in checkpoint:
                self.structural_weights = checkpoint[alt_key].numpy()
                print(f"✓ Loaded from alternative key: {self.structural_weights.shape}")

        self.model = checkpoint

    def load_feature_cache(self):
        """Load cached structural features to compute statistics."""
        print(f"\nLoading feature cache from: {self.features_cache_path}")

        with open(self.features_cache_path, 'rb') as f:
            cache = pickle.load(f)

        # Cache structure: dict with 'tc_history', 'feature_means', 'feature_stds'
        self.feature_stats = {
            'means': cache.get('feature_means'),
            'stds': cache.get('feature_stds'),
            'medians': cache.get('feature_medians', None),
        }

        print(f"✓ Loaded feature statistics")
        print(f"  Means shape: {self.feature_stats['means'].shape if self.feature_stats['means'] is not None else 'N/A'}")
        print(f"  Stds shape: {self.feature_stats['stds'].shape if self.feature_stats['stds'] is not None else 'N/A'}")

    def analyze_weight_magnitude(self) -> np.ndarray:
        """
        Analyze feature importance based on weight magnitude.

        Higher magnitude weights indicate more important features.

        Returns:
            importance_scores: [29] array of importance scores
        """
        print("\n" + "="*70)
        print("METHOD 1: WEIGHT MAGNITUDE ANALYSIS")
        print("="*70)

        # Compute L2 norm of weights for each input feature
        # structural_weights: [hidden_dim, 29]
        # For each of 29 features, compute norm across all hidden units
        importance = np.linalg.norm(self.structural_weights, axis=0)  # [29]

        # Normalize to [0, 1]
        importance_normalized = importance / importance.max()

        # Print top features
        indices = np.argsort(importance_normalized)[::-1]
        print("\nTop 15 Features by Weight Magnitude:")
        for i, idx in enumerate(indices[:15], 1):
            print(f"  {i:2d}. {FEATURE_NAMES[idx]:30s} → {importance_normalized[idx]:.4f}")

        return importance_normalized

    def analyze_feature_variance(self) -> np.ndarray:
        """
        Analyze feature importance based on variance.

        Features with higher variance are more informative.

        Returns:
            importance_scores: [29] array of importance scores
        """
        print("\n" + "="*70)
        print("METHOD 2: FEATURE VARIANCE ANALYSIS")
        print("="*70)

        stds = self.feature_stats['stds']

        # Use coefficient of variation (CV = std / mean) to account for scale
        means = self.feature_stats['means']
        cv = np.abs(stds / (means + 1e-8))  # Add epsilon to avoid division by zero

        # Normalize to [0, 1]
        importance_normalized = cv / cv.max()

        # Print top features
        indices = np.argsort(importance_normalized)[::-1]
        print("\nTop 15 Features by Variance (CV):")
        for i, idx in enumerate(indices[:15], 1):
            print(f"  {i:2d}. {FEATURE_NAMES[idx]:30s} → CV={importance_normalized[idx]:.4f} (std={stds[idx]:.4f}, mean={means[idx]:.4f})")

        return importance_normalized

    def compute_combined_ranking(self,
                                 weight_scores: np.ndarray,
                                 variance_scores: np.ndarray,
                                 weight_weight: float = 0.7,
                                 variance_weight: float = 0.3) -> np.ndarray:
        """
        Combine multiple importance scores with weights.

        Args:
            weight_scores: Importance from weight magnitude
            variance_scores: Importance from feature variance
            weight_weight: Weight for weight_scores (default: 0.7)
            variance_weight: Weight for variance_scores (default: 0.3)

        Returns:
            combined_scores: [29] array of combined importance scores
        """
        print("\n" + "="*70)
        print("COMBINED RANKING")
        print("="*70)
        print(f"Weights: {weight_weight:.1f} (magnitude) + {variance_weight:.1f} (variance)")

        combined = weight_weight * weight_scores + variance_weight * variance_scores

        # Normalize to [0, 1]
        combined_normalized = combined / combined.max()

        # Print top features
        indices = np.argsort(combined_normalized)[::-1]
        print("\n🏆 TOP 10 FEATURES (RECOMMENDED):")
        print("-" * 70)
        for i, idx in enumerate(indices[:10], 1):
            print(f"  {i:2d}. {FEATURE_NAMES[idx]:30s} → Score: {combined_normalized[idx]:.4f}")
            print(f"      Weight: {weight_scores[idx]:.4f} | Variance: {variance_scores[idx]:.4f}")

        print("\n" + "="*70)

        return combined_normalized, indices

    def plot_importance(self, combined_scores: np.ndarray, indices: np.ndarray,
                       output_path: str = 'results/experiment_05_expanded_features/feature_importance.png'):
        """
        Create visualization of feature importance.

        Args:
            combined_scores: Combined importance scores
            indices: Sorted indices (most to least important)
            output_path: Where to save the plot
        """
        print(f"\nCreating visualization...")

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Top 15 features bar chart
        ax = axes[0]
        top_15_idx = indices[:15]
        top_15_scores = combined_scores[top_15_idx]
        top_15_names = [FEATURE_NAMES[i] for i in top_15_idx]

        bars = ax.barh(range(15), top_15_scores, color='steelblue')
        ax.set_yticks(range(15))
        ax.set_yticklabels(top_15_names)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Top 15 Most Important Features (Combined Ranking)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Highlight top 10
        for i in range(10):
            bars[i].set_color('darkgreen')

        # Plot 2: All features heatmap
        ax = axes[1]
        scores_2d = combined_scores.reshape(1, -1)

        im = ax.imshow(scores_2d, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_yticks([])
        ax.set_xticks(range(29))
        ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=8)
        ax.set_title('Importance Score Heatmap (All 29 Features)', fontsize=14, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.15)
        cbar.set_label('Importance Score', fontsize=10)

        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {output_path}")

    def generate_report(self, combined_scores: np.ndarray, indices: np.ndarray,
                       output_path: str = 'results/experiment_05_expanded_features/FEATURE_SELECTION_REPORT.md'):
        """Generate markdown report with recommendations."""

        top_10_idx = indices[:10]

        report = f"""# 🔬 Feature Importance Analysis Report

**Experiment**: 05 (29 Structural Features V2)
**Date**: 2025-11-14
**Method**: Combined Weight Magnitude + Feature Variance Analysis

---

## 🏆 Top 10 Recommended Features

| Rank | Feature Name | Importance | Type | Description |
|------|--------------|------------|------|-------------|
"""

        for i, idx in enumerate(top_10_idx, 1):
            feature_type = "Phylogenetic" if idx < 20 else "Structural"
            report += f"| {i} | **{FEATURE_NAMES[idx]}** | {combined_scores[idx]:.4f} | {feature_type} | - |\n"

        report += f"""
---

## 📊 Analysis Summary

### Methodology

**Weight Magnitude Analysis (70%)**:
- Analyzed L2 norm of first-layer weights in structural stream
- Features with higher weight magnitudes have stronger influence on predictions
- Indicates which features the model learned to rely on most

**Feature Variance Analysis (30%)**:
- Computed coefficient of variation (CV = std/mean) for each feature
- Features with higher variance provide more discriminative power
- Low-variance features add little information

**Combined Score**: `0.7 * weight_magnitude + 0.3 * variance`

### Key Findings

**Feature Categories in Top 10**:
"""

        # Count categories
        phylo_count = sum(1 for idx in top_10_idx if idx < 20)
        struct_count = 10 - phylo_count

        report += f"""
- Phylogenetic Features: {phylo_count}/10 ({phylo_count*10}%)
- Structural Features: {struct_count}/10 ({struct_count*10}%)

**Feature Types Represented**:
"""

        # Categorize features
        rate_features = ['failure_rate', 'recent_failure_rate', 'flakiness_rate',
                        'execution_frequency', 'very_recent_failure_rate',
                        'medium_term_failure_rate', 'stability_score']
        count_features = ['execution_count', 'failure_count', 'pass_count',
                         'consecutive_failures', 'consecutive_passes',
                         'max_consecutive_failures', 'commit_count', 'cr_count']
        age_features = ['test_age', 'last_failure_age', 'last_pass_age', 'builds_since_change']
        trend_features = ['failure_trend', 'acceleration', 'deceleration_factor']

        selected_names = [FEATURE_NAMES[idx] for idx in top_10_idx]

        rates = sum(1 for f in selected_names if f in rate_features)
        counts = sum(1 for f in selected_names if f in count_features)
        ages = sum(1 for f in selected_names if f in age_features)
        trends = sum(1 for f in selected_names if f in trend_features)

        report += f"""
- Rate Features: {rates}/10
- Count Features: {counts}/10
- Age Features: {ages}/10
- Trend Features: {trends}/10

---

## 📋 Recommended Actions

### ✅ Next Step: Experiment 06 - Feature Selection

**Configuration**:
```yaml
structural:
  input_dim: 10  # ← Reduced from 29 to 10
  extractor:
    use_v2_5: true  # ← New V2.5 with selected features
    selected_features: {list(top_10_idx)}
```

**Expected Impact**:
- ✅ Reduce overfitting (fewer features)
- ✅ Maintain signal from most important features
- ✅ Target APFD: 0.62-0.65 (match or beat 04a baseline)

### 🎯 Success Criteria for Exp 06

| Metric | Target | Baseline (04a) |
|--------|--------|----------------|
| **APFD** | ≥ 0.62 | 0.6210 |
| **F1 Macro** | ≥ 0.52 | 0.5294 |
| **APFD ≥ 0.7** | ≥ 40% | 40.8% |

---

## 📝 Full Feature Ranking

| Rank | Feature | Score | Weight Mag | Variance |
|------|---------|-------|------------|----------|
"""

        for i, idx in enumerate(indices, 1):
            report += f"| {i} | {FEATURE_NAMES[idx]} | {combined_scores[idx]:.4f} | - | - |\n"

        report += """
---

**Generated by**: Feature Importance Analyzer
**Model**: results/experiment_05_expanded_features/best_model.pt
"""

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)

        print(f"✓ Saved report to: {output_path}")

        return top_10_idx


def main():
    """Main analysis pipeline."""

    print("="*70)
    print("FEATURE IMPORTANCE ANALYSIS - EXPERIMENT 05")
    print("="*70)

    # Paths
    model_path = "results/experiment_05_expanded_features/best_model.pt"
    cache_path = "cache/structural_features_v2.pkl"

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Please ensure Experiment 05 has completed successfully.")
        return

    if not os.path.exists(cache_path):
        print(f"❌ Feature cache not found: {cache_path}")
        return

    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(model_path, cache_path)

    # Load model and data
    analyzer.load_model()
    analyzer.load_feature_cache()

    # Method 1: Weight magnitude
    weight_scores = analyzer.analyze_weight_magnitude()

    # Method 2: Feature variance
    variance_scores = analyzer.analyze_feature_variance()

    # Combined ranking
    combined_scores, indices = analyzer.compute_combined_ranking(
        weight_scores,
        variance_scores,
        weight_weight=0.7,
        variance_weight=0.3
    )

    # Visualization
    analyzer.plot_importance(combined_scores, indices)

    # Generate report
    top_10_idx = analyzer.generate_report(combined_scores, indices)

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n🏆 Top 10 Selected Feature Indices:")
    print(f"   {list(top_10_idx)}")
    print(f"\n📄 Reports saved to: results/experiment_05_expanded_features/")
    print(f"   - feature_importance.png")
    print(f"   - FEATURE_SELECTION_REPORT.md")
    print("\n🚀 Next Step: Create Experiment 06 with these 10 features")


if __name__ == "__main__":
    main()
