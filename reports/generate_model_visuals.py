"""
Generate interactive HTML visualizations explaining the Filo-Priori pipeline.

Outputs a single HTML file with:
1) Pipeline overview (Sankey)
2) Graph edge composition (from config)
3) Threshold search settings
4) APFD distribution (if results CSV exists)
5) Orphan score distribution (if results CSV exists)

Usage:
    python reports/generate_model_visuals.py \
        --config configs/experiment_industry_optimized_v3.yaml \
        --results-dir results/experiment_industry_optimized_v3 \
        --output reports/model_explainer.html
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly import io as pio


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def pipeline_sankey() -> go.Figure:
    """High-level pipeline view."""
    labels = [
        "Raw Data", "Text/Commits", "Structural History",
        "SBERT Embeddings", "Structural Features",
        "Multi-Edge Graph", "Semantic Stream", "Structural GAT",
        "Fusion", "Classifier", "Threshold Search", "Orphan KNN",
        "Hybrid Ranking", "APFD Evaluation"
    ]
    links = dict(
        source=[0, 0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11],
        target=[1, 2, 3, 4, 5, 5, 6, 7, 7, 8, 8, 9, 10, 10, 11, 12],
        value=[5, 5, 5, 5, 4, 3, 5, 5, 4, 4, 4, 4, 3, 3, 4, 4],
        color=["#1f77b4", "#1f77b4", "#9467bd", "#17becf", "#8c564b",
               "#bcbd22", "#17becf", "#8c564b", "#ff7f0e", "#2ca02c",
               "#2ca02c", "#e377c2", "#d62728", "#d62728", "#7f7f7f", "#7f7f7f"]
    )
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, pad=20, thickness=20, color="#cccccc"),
        link=links
    ))
    fig.update_layout(title="Pipeline Overview")
    return fig


def graph_edge_bar(config: dict) -> go.Figure:
    graph_cfg = config.get("graph", {})
    weights = graph_cfg.get("edge_weights", {})
    edge_types = list(weights.keys()) or ["co_failure", "co_success", "semantic", "temporal", "component"]
    vals = [weights.get(e, 0) for e in edge_types]

    fig = go.Figure(go.Bar(x=edge_types, y=vals, marker_color="#1f77b4"))
    fig.update_layout(
        title="Graph Edge Weights",
        xaxis_title="Edge Type",
        yaxis_title="Weight",
        yaxis=dict(range=[0, max(vals + [1])])
    )
    return fig


def threshold_settings_fig(config: dict) -> go.Figure:
    eval_cfg = config.get("evaluation", {}).get("threshold_search", {})
    coarse = eval_cfg.get("coarse_step", eval_cfg.get("step", 0.02))
    fine = eval_cfg.get("fine_step", coarse / 2)
    beta = eval_cfg.get("beta", 1.0)
    two_phase = eval_cfg.get("two_phase", False)
    opt_for = eval_cfg.get("optimize_for", "f1_macro")

    labels = ["coarse_step", "fine_step", "beta"]
    values = [coarse, fine, beta]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#ff7f0e"))
    fig.update_layout(
        title=f"Threshold Search Settings ({'two-phase' if two_phase else 'single-phase'}) — optimize_for={opt_for}",
        yaxis_title="Value"
    )
    return fig


def apfd_distribution(results_dir: Path) -> go.Figure:
    apfd_path = results_dir / "apfd_per_build_FULL_testcsv.csv"
    df = _safe_read_csv(apfd_path)
    if df is None or "APFD" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="APFD results not found", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title="APFD Distribution (missing data)")
        return fig

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df["APFD"], nbinsx=20, marker_color="#2ca02c"))
    fig.update_layout(
        title=f"APFD Distribution (n={len(df)})",
        xaxis_title="APFD",
        yaxis_title="Build count",
        bargap=0.05
    )
    return fig


def orphan_distribution(results_dir: Path) -> go.Figure:
    priority_path = results_dir / "prioritized_test_cases_FULL_testcsv.csv"
    df = _safe_read_csv(priority_path)
    if df is None or "hybrid_score" not in df.columns or "probability" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Orphan scores not found", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title="Orphan Score Distribution (missing data)")
        return fig

    orphan_mask = df["probability"].sub(0.5).abs() < 1e-3
    orphan_scores = df.loc[orphan_mask, "hybrid_score"]
    fig = go.Figure()
    fig.add_trace(go.Box(y=orphan_scores, name="Orphans", boxmean=True, marker_color="#e377c2"))
    fig.update_layout(
        title=f"Orphan Hybrid Scores (n={len(orphan_scores)})",
        yaxis_title="Score"
    )
    return fig


def compose_html(figs, output: Path):
    parts = []
    for fig in figs:
        parts.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))

    template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Filo-Priori Model Explainer</title>
</head>
<body>
  <h1>Filo-Priori Model Explainer</h1>
  <p>Interactive visuals covering pipeline, graph design, thresholds, and evaluation.</p>
  {''.join(parts)}
</body>
</html>
"""
    output.write_text(template, encoding="utf-8")
    print(f"✅ Visualization saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Generate model visualization HTML.")
    parser.add_argument("--config", default="configs/experiment_industry_optimized_v3.yaml")
    parser.add_argument("--results-dir", default="results/experiment_industry_optimized_v3")
    parser.add_argument("--output", default="reports/model_explainer.html")
    args = parser.parse_args()

    config_path = Path(args.config)
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    figs = [
        pipeline_sankey(),
        graph_edge_bar(config),
        threshold_settings_fig(config),
        apfd_distribution(results_dir),
        orphan_distribution(results_dir)
    ]

    compose_html(figs, output_path)


if __name__ == "__main__":
    main()
