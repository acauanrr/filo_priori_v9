#!/usr/bin/env python3
"""
Interactive Multi-Edge Phylogenetic Graph Visualization

Generates an interactive HTML visualization of the phylogenetic graph showing
co-failure, co-success, and semantic relationships between test cases.

Uses PyVis for interactive network visualization.

Author: Filo-Priori V8 Team
Date: 2025-11-14
"""

import pickle
import numpy as np
from pyvis.network import Network
from collections import defaultdict, Counter
import os

print("\n" + "="*70)
print("INTERACTIVE MULTI-EDGE PHYLOGENETIC GRAPH VISUALIZATION")
print("="*70)

# Load graph from cache
cache_path = "cache/multi_edge_graph.pkl"
print(f"\n[1/6] Loading graph from cache: {cache_path}")

with open(cache_path, 'rb') as f:
    graph_data = pickle.load(f)

print(f"✓ Graph data loaded")
print(f"  Keys: {list(graph_data.keys())}")

# Extract data
tc_to_idx = graph_data['tc_to_idx']
idx_to_tc = graph_data['idx_to_tc']
edges_multi = graph_data.get('edges_multi', {})

# Edge types configuration
edge_type_config = {
    'co_failure': {
        'weight_key': 'co_failure',
        'color': '#e74c3c',  # Red
        'label': 'Co-Failure',
        'width_multiplier': 1.5,  # REDUCED for better visibility
        'description': 'Tests that failed together'
    },
    'co_success': {
        'weight_key': 'co_success',
        'color': '#27ae60',  # Green
        'label': 'Co-Success',
        'width_multiplier': 0.8,  # REDUCED for better visibility
        'description': 'Tests that passed together'
    },
    'semantic': {
        'weight_key': 'semantic',
        'color': '#3498db',  # Blue
        'label': 'Semantic',
        'width_multiplier': 0.6,  # REDUCED for better visibility
        'description': 'Semantically similar tests'
    }
}

num_nodes = len(tc_to_idx)
print(f"\n[2/6] Processing graph structure")
print(f"  Total test cases (nodes): {num_nodes}")

# Process edges by type
edges_by_type = {
    'co_failure': [],
    'co_success': [],
    'semantic': []
}

# Extract edges from edges_multi dictionary
# Structure: edges_multi[(src, dst)] = {'co_failure': weight, 'co_success': weight, 'semantic': weight}
for (src, dst), edge_types_dict in edges_multi.items():
    if isinstance(edge_types_dict, dict):
        for edge_type, weight in edge_types_dict.items():
            if edge_type in edges_by_type and weight > 0:
                # Convert weight to Python float (from numpy float32)
                edges_by_type[edge_type].append((src, dst, float(weight)))

print(f"\n  Edge counts by type:")
for edge_type, edges in edges_by_type.items():
    print(f"    {edge_type_config[edge_type]['label']:12s}: {len(edges):6d} edges")

total_edges = sum(len(edges) for edges in edges_by_type.values())
print(f"  Total edges: {total_edges}")

# Select representative sample for visualization
print(f"\n[3/6] Selecting representative sample for visualization")

# Strategy: Select nodes with highest degree (most connected)
node_degree = defaultdict(int)
for edge_type, edges in edges_by_type.items():
    for src, dst, weight in edges:
        node_degree[src] += 1
        node_degree[dst] += 1

# Sort by degree
sorted_nodes = sorted(node_degree.items(), key=lambda x: x[1], reverse=True)

# Select top N most connected nodes
SAMPLE_SIZE = min(80, len(sorted_nodes))  # Increased for better representation
sample_nodes = set([node for node, degree in sorted_nodes[:SAMPLE_SIZE]])

print(f"  Selected {len(sample_nodes)} highly connected test cases")
if len(sorted_nodes) > 0:
    max_degree = sorted_nodes[0][1]
    min_degree = sorted_nodes[min(SAMPLE_SIZE-1, len(sorted_nodes)-1)][1]
    print(f"  Degree range: {max_degree} (max) to {min_degree} (min)")
else:
    print("  No edges found in graph")

# Filter edges to only include sampled nodes
sampled_edges_by_type = {}
for edge_type, edges in edges_by_type.items():
    sampled_edges = [
        (src, dst, weight)
        for src, dst, weight in edges
        if src in sample_nodes and dst in sample_nodes
    ]
    sampled_edges_by_type[edge_type] = sampled_edges
    print(f"    {edge_type_config[edge_type]['label']:12s}: {len(sampled_edges):4d} edges (sampled)")

# Create PyVis network
print(f"\n[4/6] Creating interactive network visualization")

net = Network(
    height='900px',
    width='100%',
    bgcolor='#ffffff',
    font_color='#000000',
    heading='Multi-Edge Phylogenetic Graph: Test Case Relationships'
)

# Configure physics for better layout - OPTIMIZED FOR STABILITY
net.set_options("""
var options = {
  "physics": {
    "enabled": true,
    "forceAtlas2Based": {
      "gravitationalConstant": -80,
      "centralGravity": 0.005,
      "springLength": 250,
      "springConstant": 0.02,
      "damping": 0.95,
      "avoidOverlap": 1.0
    },
    "maxVelocity": 10,
    "minVelocity": 0.75,
    "solver": "forceAtlas2Based",
    "timestep": 0.5,
    "stabilization": {
      "enabled": true,
      "iterations": 2000,
      "updateInterval": 50,
      "onlyDynamicEdges": false,
      "fit": true
    }
  },
  "nodes": {
    "font": {
      "size": 10,
      "face": "arial"
    },
    "borderWidth": 1,
    "borderWidthSelected": 2
  },
  "edges": {
    "smooth": {
      "type": "continuous",
      "roundness": 0.5
    }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100,
    "navigationButtons": true,
    "keyboard": true,
    "dragNodes": true,
    "dragView": true,
    "zoomView": true
  }
}
""")

# Add nodes
print(f"  Adding {len(sample_nodes)} nodes...")

for node_idx in sample_nodes:
    # Get test case name
    tc_name = idx_to_tc.get(node_idx, f"TC_{node_idx}")

    # Shorten name for display
    display_name = tc_name.split('::')[-1] if '::' in tc_name else tc_name
    if len(display_name) > 30:
        display_name = display_name[:27] + "..."

    # Count connections by type
    co_failure_count = sum(1 for src, dst, _ in sampled_edges_by_type['co_failure']
                           if src == node_idx or dst == node_idx)
    co_success_count = sum(1 for src, dst, _ in sampled_edges_by_type['co_success']
                           if src == node_idx or dst == node_idx)
    semantic_count = sum(1 for src, dst, _ in sampled_edges_by_type['semantic']
                         if src == node_idx or dst == node_idx)

    total_connections = co_failure_count + co_success_count + semantic_count

    # Node size based on degree (REDUCED for better visibility)
    node_size = 5 + min(total_connections * 0.3, 15)

    # Node color based on dominant edge type
    if co_failure_count > max(co_success_count, semantic_count):
        node_color = '#ffcccc'  # Light red (failure-prone)
    elif co_success_count > semantic_count:
        node_color = '#ccffcc'  # Light green (stable)
    else:
        node_color = '#cce5ff'  # Light blue (semantic)

    # Create tooltip
    tooltip = f"""
    <b>{tc_name}</b><br>
    <hr>
    <b>Connections:</b><br>
    • Co-Failures: {co_failure_count}<br>
    • Co-Successes: {co_success_count}<br>
    • Semantic: {semantic_count}<br>
    <b>Total:</b> {total_connections}
    """

    net.add_node(
        int(node_idx),  # Convert to Python int
        label=display_name,
        title=tooltip,
        size=node_size,
        color=node_color,
        borderWidth=2,
        borderWidthSelected=4
    )

# Add edges by type
print(f"  Adding edges by type...")

for edge_type, edges in sampled_edges_by_type.items():
    config = edge_type_config[edge_type]

    for src, dst, weight in edges:
        # Edge width based on weight
        edge_width = config['width_multiplier'] * (1 + weight * 3)

        # Create tooltip
        edge_tooltip = f"""
        <b>{config['label']} Edge</b><br>
        Weight: {weight:.3f}<br>
        {config['description']}
        """

        net.add_edge(
            int(src),  # Convert to Python int
            int(dst),  # Convert to Python int
            color=config['color'],
            width=edge_width,
            title=edge_tooltip,
            label=f"{weight:.2f}" if weight > 0.5 else ""  # Show label for strong connections
        )

print(f"✓ Network visualization created")

# Generate legend HTML
legend_html = """
<div style="position: fixed; top: 80px; right: 20px; background: white;
            padding: 15px; border: 2px solid #333; border-radius: 10px;
            font-family: Arial; font-size: 14px; z-index: 1000; max-width: 250px;">
    <h3 style="margin-top: 0; color: #333;">Legend</h3>

    <h4 style="margin-bottom: 5px; color: #555;">Edge Types:</h4>
    <div style="margin-bottom: 5px;">
        <span style="display: inline-block; width: 20px; height: 3px;
                     background: #e74c3c; vertical-align: middle;"></span>
        <b style="color: #e74c3c;"> Co-Failure</b> - Tests failing together
    </div>
    <div style="margin-bottom: 5px;">
        <span style="display: inline-block; width: 20px; height: 3px;
                     background: #27ae60; vertical-align: middle;"></span>
        <b style="color: #27ae60;"> Co-Success</b> - Tests passing together
    </div>
    <div style="margin-bottom: 15px;">
        <span style="display: inline-block; width: 20px; height: 3px;
                     background: #3498db; vertical-align: middle;"></span>
        <b style="color: #3498db;"> Semantic</b> - Similar test content
    </div>

    <h4 style="margin-bottom: 5px; color: #555;">Node Colors:</h4>
    <div style="margin-bottom: 3px;">
        <span style="display: inline-block; width: 15px; height: 15px;
                     background: #ffcccc; border: 1px solid #333;"></span>
        Failure-prone tests
    </div>
    <div style="margin-bottom: 3px;">
        <span style="display: inline-block; width: 15px; height: 15px;
                     background: #ccffcc; border: 1px solid #333;"></span>
        Stable tests
    </div>
    <div style="margin-bottom: 15px;">
        <span style="display: inline-block; width: 15px; height: 15px;
                     background: #cce5ff; border: 1px solid #333;"></span>
        Semantic-linked tests
    </div>

    <p style="font-size: 12px; color: #666; margin-top: 10px; margin-bottom: 0;">
        <b>Node size</b> = connection count<br>
        <b>Edge width</b> = relationship strength<br>
        <b>Hover</b> for details
    </p>
</div>
"""

# Save to HTML
print(f"\n[5/6] Saving interactive visualization...")

output_path = "results/publication/multi_edge_phylogenetic_graph_interactive.html"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

net.save_graph(output_path)

# Add legend to HTML
with open(output_path, 'r') as f:
    html_content = f.read()

# Insert legend before </body>
html_content = html_content.replace('</body>', f'{legend_html}</body>')

with open(output_path, 'w') as f:
    f.write(html_content)

print(f"✓ Interactive visualization saved: {output_path}")

# Generate statistics report
print(f"\n[6/6] Generating statistics report")

stats_report = f"""
# Multi-Edge Phylogenetic Graph Statistics

**Generated**: 2025-11-14
**Graph Type**: Multi-Edge Directed Graph

## Graph Composition

**Total Nodes**: {num_nodes} test cases
**Visualized Nodes**: {len(sample_nodes)} (top {SAMPLE_SIZE} most connected)

**Total Edges**: {total_edges}
- Co-Failure: {len(edges_by_type['co_failure'])} ({len(edges_by_type['co_failure'])/total_edges*100:.1f}%)
- Co-Success: {len(edges_by_type['co_success'])} ({len(edges_by_type['co_success'])/total_edges*100:.1f}%)
- Semantic: {len(edges_by_type['semantic'])} ({len(edges_by_type['semantic'])/total_edges*100:.1f}%)

**Sampled Edges**: {sum(len(e) for e in sampled_edges_by_type.values())}
- Co-Failure: {len(sampled_edges_by_type['co_failure'])}
- Co-Success: {len(sampled_edges_by_type['co_success'])}
- Semantic: {len(sampled_edges_by_type['semantic'])}

## Node Statistics

**Most Connected Nodes**:
"""

for i, (node, degree) in enumerate(sorted_nodes[:10], 1):
    tc_name = idx_to_tc.get(node, f"TC_{node}")
    stats_report += f"{i}. {tc_name} - {degree} connections\n"

stats_report += f"""

## Edge Type Descriptions

### Co-Failure Edges (Red)
- **Count**: {len(edges_by_type['co_failure'])}
- **Description**: Tests that failed together in the same build
- **Signal**: Direct failure correlation
- **Weight**: count(fail_together) / count(occur_together)

### Co-Success Edges (Green)
- **Count**: {len(edges_by_type['co_success'])}
- **Description**: Tests that passed together in the same build
- **Signal**: Complementary stability pattern
- **Weight**: count(pass_together) / count(occur_together) × 0.5

### Semantic Edges (Blue)
- **Count**: {len(edges_by_type['semantic'])}
- **Description**: Tests with similar semantic embeddings
- **Signal**: Content-based relationships
- **Weight**: cosine_similarity(embedding_u, embedding_v) × 0.3

## Visualization Features

**Interactive Controls**:
- Drag nodes to rearrange
- Scroll to zoom in/out
- Click node to highlight connections
- Hover for detailed information

**Color Coding**:
- **Pink nodes**: Failure-prone (many co-failure edges)
- **Green nodes**: Stable (many co-success edges)
- **Blue nodes**: Semantic-linked (many semantic edges)

**Edge Width**: Proportional to relationship strength
**Node Size**: Proportional to number of connections

## File Locations

**Interactive Visualization**: {output_path}
**Graph Cache**: {cache_path}

---

**Usage**: Open {output_path} in a web browser to explore the graph interactively.
"""

stats_path = "results/publication/multi_edge_graph_statistics.md"
with open(stats_path, 'w') as f:
    f.write(stats_report)

print(f"✓ Statistics report saved: {stats_path}")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print(f"\n📊 Interactive Graph: {output_path}")
print(f"📋 Statistics: {stats_path}")
print(f"\n🌐 Open {output_path} in a web browser to explore!")
print("="*70)
