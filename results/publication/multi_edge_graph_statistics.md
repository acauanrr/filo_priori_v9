
# Multi-Edge Phylogenetic Graph Statistics

**Generated**: 2025-11-14
**Graph Type**: Multi-Edge Directed Graph

## Graph Composition

**Total Nodes**: 2347 test cases
**Visualized Nodes**: 80 (top 80 most connected)

**Total Edges**: 461493
- Co-Failure: 495 (0.1%)
- Co-Success: 207913 (45.1%)
- Semantic: 253085 (54.8%)

**Sampled Edges**: 2056
- Co-Failure: 6
- Co-Success: 1660
- Semantic: 390

## Node Statistics

**Most Connected Nodes**:
1. MCA-1015 - 48297 connections
2. MCA-101956 - 48145 connections
3. MCA-1012 - 48125 connections
4. MCA-1011 - 48117 connections
5. MCA-1013 - 48098 connections
6. MCA-1037 - 2630 connections
7. MCA-101960 - 2562 connections
8. MCA-101962 - 2562 connections
9. MCA-103639 - 2551 connections
10. MCA-101961 - 2527 connections


## Edge Type Descriptions

### Co-Failure Edges (Red)
- **Count**: 495
- **Description**: Tests that failed together in the same build
- **Signal**: Direct failure correlation
- **Weight**: count(fail_together) / count(occur_together)

### Co-Success Edges (Green)
- **Count**: 207913
- **Description**: Tests that passed together in the same build
- **Signal**: Complementary stability pattern
- **Weight**: count(pass_together) / count(occur_together) × 0.5

### Semantic Edges (Blue)
- **Count**: 253085
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

**Interactive Visualization**: results/publication/multi_edge_phylogenetic_graph_interactive.html
**Graph Cache**: cache/multi_edge_graph.pkl

---

**Usage**: Open results/publication/multi_edge_phylogenetic_graph_interactive.html in a web browser to explore the graph interactively.
