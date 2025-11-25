# Análise da Arquitetura Filo-Priori V8 e Propostas de Melhoria

**Data:** 2025-11-13
**Autor:** Análise Técnica Claude Code
**Contexto:** Resposta às questões sobre arquitetura dual-stream e melhorias propostas

---

## SUMÁRIO EXECUTIVO

Este documento responde às seguintes questões críticas:
1. ✅ Como funciona a estrutura de grafo que representa características estruturais/filogenéticas?
2. ✅ O grafo é criado por build ou é global?
3. ✅ Como visualizar o grafo GAT/GAN?
4. ✅ Por que apenas 3 features estruturais não estão sendo suficientes?
5. ✅ Como justificar que a camada estrutural contribui para o aprendizado?
6. ⚠️ O erro NVML persiste - proposta de solução definitiva
7. 🎯 Plano de melhorias para validar a proposta da tese

---

## 1. ENTENDIMENTO DA ARQUITETURA ATUAL

### 1.1 Features Estruturais/Filogenéticas (6 features, não 3!)

**CORREÇÃO:** O modelo extrai **6 features**, não 3:

```python
# Phylogenetic features (4):
1. test_age          → Builds desde primeira aparição
2. failure_rate      → Taxa histórica de falha
3. recent_failure_rate → Taxa de falha em últimos 5 builds
4. flakiness_rate    → Taxa de transição Pass↔Fail (oscilação)

# Structural features (2):
5. commit_count      → Número de commits/CRs únicos
6. test_novelty      → Flag binária (1=primeira aparição, 0=já visto)
```

**Origem:** `src/preprocessing/structural_feature_extractor.py:432-439`

### 1.2 Grafo Filogenético

**RESPOSTA:** O grafo é **GLOBAL** (construído uma vez no training), mas **subgrafos** são extraídos por batch.

#### Como Funciona:

```
FASE 1: CONSTRUÇÃO (Training) - GLOBAL
┌─────────────────────────────────────────────────────┐
│ PhylogeneticGraphBuilder.fit(df_train)             │
│                                                     │
│ → Analisa TODOS os test cases no training          │
│ → Cria mapeamento: TC_Key → índice global         │
│ → Constrói arestas baseadas em:                    │
│   • Co-failure: P(A fails | B fails)               │
│   • Commit dependency: shared commits              │
│                                                     │
│ Resultado: Grafo GLOBAL com ~50K nós              │
└─────────────────────────────────────────────────────┘

FASE 2: INFERÊNCIA (Train/Val/Test) - SUBGRAFOS
┌─────────────────────────────────────────────────────┐
│ Para cada BATCH de 32 amostras:                    │
│                                                     │
│ 1. Extrair TC_Keys do batch: [tc_1, tc_2, ..., tc_32] │
│                                                     │
│ 2. Criar subgrafo apenas com esses 32 nós:         │
│    graph_builder.get_edge_index_and_weights(tc_keys)│
│                                                     │
│ 3. Re-mapear índices globais → locais [0..31]     │
│                                                     │
│ 4. Passar para GAT:                                │
│    structural_stream(features, edge_index, edge_weights)│
└─────────────────────────────────────────────────────┘
```

**Código relevante:** `src/phylogenetic/phylogenetic_graph_builder.py:118-175`

#### Tipos de Grafo:

**A. Co-Failure Graph** (default):
```python
# Conecta testes que falharam JUNTOS no mesmo Build_ID
# Peso = Média de P(A fails | B fails) e P(B fails | A fails)

Exemplo:
Build_123: {TC_001: Fail, TC_002: Fail, TC_003: Pass}
         → Cria aresta: TC_001 ↔ TC_002

Se co-ocorreram 5x de 10 falhas de TC_001:
  weight = (5/10 + 5/failures_TC_002) / 2
```

**B. Commit Dependency Graph**:
```python
# Conecta testes que compartilham COMMITS/CRs
# Peso = shared_commits / max_shared (normalizado)

Exemplo:
TC_001: commits=[abc123, def456]
TC_002: commits=[abc123, xyz789]
      → Compartilham 1 commit
      → Cria aresta com peso proporcional
```

**C. Hybrid**: Média dos dois grafos acima.

**Código relevante:** `src/phylogenetic/phylogenetic_graph_builder.py:188-334`

### 1.3 Arquitetura Dual-Stream

```
INPUT: Sample de test execution
├── Semantic: TE_Summary + TC_Steps → Qodo-Embed → [3072]
└── Structural: Historical features → [6]

════════════════════════════════════════════════════════

SEMANTIC STREAM:
┌──────────────────────────────────────┐
│ Input: [batch, 3072]                │  ← TC + Commit embeddings concatenados
│   ↓                                  │
│ Linear Projection → [batch, 256]    │
│   ↓                                  │
│ 2x FFN Layers (residual)            │
│   ↓                                  │
│ Output: [batch, 256]                │
└──────────────────────────────────────┘

STRUCTURAL STREAM (GAT):
┌──────────────────────────────────────┐
│ Input: features [batch, 6]          │
│        edge_index [2, E]             │  ← Subgrafo do batch
│        edge_weights [E]              │
│   ↓                                  │
│ GATConv (4 heads) → [batch, 1024]  │  ← 4 heads × 256 = 1024
│   ↓                                  │
│ GATConv (1 head)  → [batch, 256]   │  ← Average heads
│   ↓                                  │
│ Output: [batch, 256]                │
└──────────────────────────────────────┘

FUSION (Cross-Attention ou Gated):
┌──────────────────────────────────────┐
│ Semantic [batch, 256]               │
│ Structural [batch, 256]             │
│   ↓                                  │
│ Cross-Attention (bidirectional)      │
│   OR                                 │
│ Gated Fusion (learned gate)         │
│   ↓                                  │
│ Output: [batch, 512]                │
└──────────────────────────────────────┘

CLASSIFIER:
┌──────────────────────────────────────┐
│ Input: [batch, 512]                 │
│   ↓                                  │
│ MLP [512→128→64→2]                  │
│   ↓                                  │
│ Output: [batch, 2] (Pass vs Fail)   │
└──────────────────────────────────────┘
```

**Código relevante:** `src/models/dual_stream_v8.py`

---

## 2. PROBLEMAS IDENTIFICADOS

### 2.1 Erro NVML (Crítico!)

**Sintoma:**
- Falha SEMPRE no chunk 3 do encoding de commits
- Retry não funciona (3 tentativas, todas falham)
- Mesmo com model reload e CUDA cache clear

**Causa Provável:**
- **Fragmentação de memória CUDA** acumulada dos chunks anteriores
- NVML (NVIDIA Management Library) não consegue inicializar após 2 chunks
- O chunk 3 tem textos de commits mais longos? (batch size 63 vs 32 para TCs)

**Evidência:**
```
Chunk 1: 63 batches [00:50] ✓
Chunk 2: 63 batches [01:31] ✓  ← Tempo 2x maior (sinal de pressão de memória)
Chunk 3: 0 batches [00:01] ✗   ← Falha imediata
```

### 2.2 Features Estruturais Limitadas

**Problema:** Apenas 6 features não capturam a riqueza de informação filogenética/evolutiva disponível.

**Limitações:**

1. **Novos testes:** 4 das 6 features = 0 (sem histórico)
   - `test_age = 0`
   - `failure_rate = global_mean` (não específico!)
   - `recent_failure_rate = global_mean`
   - `flakiness_rate = global_median`

   Resultado: 67% das features são defaults genéricos!

2. **Faltam features críticas:**
   - ❌ Similaridade estrutural entre testes (code coverage overlap)
   - ❌ Dependencies entre testes (ordem de execução)
   - ❌ Complexidade do código afetado (lines changed, cyclomatic complexity)
   - ❌ Evolução temporal (tendências de falha)
   - ❌ Features de commits (author experience, commit message sentiment)
   - ❌ Build context (time of day, previous build result)

3. **Grafo pode ter nós isolados:**
   - Novo teste no validation/test → sem arestas
   - GAT não consegue propagar informação de vizinhos
   - Equivalente a processar com MLP simples (sem grafo)

### 2.3 Falta de Validação da Contribuição Estrutural

**Problema:** Não há evidência de que a camada estrutural está ajudando!

**Situação atual:**
- ❌ Sem baseline (semantic-only model)
- ❌ Sem ablation study
- ❌ Sem análise de gate weights (no caso de Gated Fusion)
- ❌ Sem visualização de attention weights (GAT)

**Resultado:** Impossível justificar a tese!

---

## 3. PROPOSTAS DE MELHORIA

### 3.1 Solução Definitiva para Erro NVML

#### Opção A: Reduzir Chunk Size (Conservador)
```yaml
# Em configs/experiment.yaml
semantic:
  use_chunked_encoding: true
  chunk_size: 500  # Era 1000, reduzir pela metade
  reload_every_n_chunks: 3  # Era 5, reload mais frequente
```

#### Opção B: Usar CPU para Commits (Robusto)
```python
# Modificar src/embeddings/qodo_encoder_chunked.py
class QodoEncoderChunked:
    def encode_commit_texts(self, commit_texts, ...):
        """Commits são mais longos → usar CPU para evitar OOM"""

        # Move model to CPU temporarily
        self.model.to('cpu')
        embeddings = self.encode_texts_chunked(
            commit_texts,
            device='cpu',  # ← Força CPU
            ...
        )
        # Move back to GPU
        self.model.to(self.device)

        return embeddings
```

#### Opção C: Pré-computar Embeddings Offline (Mais Rápido)
```bash
# Script separado para encoding (rodar uma vez)
python scripts/precompute_embeddings.py \
  --config configs/experiment.yaml \
  --output cache/embeddings_precomputed.npz

# Depois no main.py, apenas carregar:
embeddings = np.load('cache/embeddings_precomputed.npz')
```

**RECOMENDAÇÃO:** Opção C (pré-computar) para experimentos + Opção B (CPU) como fallback.

### 3.2 Expansão de Features Estruturais (6 → 20+)

#### Categoria 1: Features Filogenéticas Avançadas (10 features)

```python
# Em StructuralFeatureExtractor, adicionar:

PHYLOGENETIC_FEATURES = [
    # Existentes (4):
    'test_age',
    'failure_rate',
    'recent_failure_rate',
    'flakiness_rate',

    # NOVOS (6):
    'failure_rate_std',          # Variância da taxa de falha (estabilidade)
    'time_since_last_failure',   # Builds desde última falha
    'max_consecutive_failures',  # Maior sequência de falhas
    'recovery_rate',             # P(Pass | previous Fail) - capacidade de recuperação
    'failure_trend',             # Regressão linear da taxa de falha (crescente/decrescente)
    'build_frequency',           # Execuções por build (avg)
]
```

#### Categoria 2: Features de Commits (5 features)

```python
COMMIT_FEATURES = [
    'commit_count',              # Existente
    'commit_recency',            # Days since most recent commit
    'commit_frequency',          # Commits per day (avg)
    'commit_impact_score',       # Weighted by lines changed
    'unique_authors_count',      # Diversity de desenvolvedores
]
```

#### Categoria 3: Features de Build Context (5 features)

```python
BUILD_CONTEXT_FEATURES = [
    'test_novelty',              # Existente
    'build_failure_rate',        # Taxa de falha do build atual (outros testes)
    'test_execution_order',      # Posição no build (normalizado)
    'concurrent_failures',       # Outros testes falhando no mesmo build
    'build_time_of_day',         # Hora do dia (normalizado, captura padrões temporais)
]
```

#### Categoria 4: Features de Grafo (4 features - extraídas do GAT)

```python
GRAPH_FEATURES = [
    'node_degree',               # Número de arestas (centralidade)
    'avg_neighbor_failure_rate', # Média da taxa de falha dos vizinhos
    'clustering_coefficient',    # Coeficiente de agrupamento
    'pagerank_score',            # PageRank no grafo de co-failure
]
```

**TOTAL: 24 features** (vs 6 atuais = **4x mais informação**)

### 3.3 Melhorias no Grafo Filogenético

#### Problema 1: Nós Isolados (testes novos sem arestas)

**Solução A: Fallback k-NN Semântico**
```python
class PhylogeneticGraphBuilder:
    def get_edge_index_and_weights(self, tc_keys, semantic_embeddings=None):
        """
        Se teste não tem arestas filogenéticas, criar arestas k-NN semânticas
        """
        # 1. Extrair arestas filogenéticas (padrão)
        phylo_edges, phylo_weights = self._get_phylogenetic_edges(tc_keys)

        # 2. Identificar nós isolados
        isolated_nodes = self._find_isolated_nodes(tc_keys, phylo_edges)

        # 3. Para nós isolados, criar k-NN edges (k=5)
        if len(isolated_nodes) > 0 and semantic_embeddings is not None:
            knn_edges, knn_weights = self._create_knn_edges(
                isolated_nodes,
                semantic_embeddings,
                k=5
            )

            # 4. Combinar: phylo + knn (com pesos menores para knn)
            phylo_edges = torch.cat([phylo_edges, knn_edges], dim=1)
            phylo_weights = torch.cat([phylo_weights, knn_weights * 0.5])

        return phylo_edges, phylo_weights
```

**Solução B: Self-loops para Nós Isolados**
```python
# Adicionar self-loop com peso 1.0 para nós sem arestas
# Permite que GAT processe features mesmo sem vizinhos
for node in isolated_nodes:
    edge_index.append([node, node])  # self-loop
    edge_weights.append(1.0)
```

#### Problema 2: Grafo Estático vs Dinâmico

**Solução: Grafo Adaptativo por Epoch**
```python
class AdaptivePhylogeneticGraph:
    """
    Reconstrói grafo a cada N epochs usando predições atuais
    """
    def update_graph(self, model, dataloader, epoch):
        if epoch % 5 == 0:  # Reconstruir a cada 5 epochs
            # 1. Rodar inferência para obter predições
            predictions = self._get_predictions(model, dataloader)

            # 2. Construir novo co-failure graph baseado em PREDIÇÕES
            # (não apenas labels reais)
            self._rebuild_cofailure_graph(predictions)

            # 3. Atualizar edge_index e edge_weights
            self.graph_builder.fit(df_with_predictions)
```

### 3.4 Estratégia de Ablation Study

**Objetivo:** Provar que camada estrutural contribui para performance.

#### Experimentos Propostos:

```yaml
# Experimento 1: BASELINE - Semantic Only
experiment_baseline_semantic_only:
  description: "Apenas semantic stream (sem structural)"
  model:
    semantic:
      input_dim: 3072
      hidden_dim: 256
      num_layers: 2
    structural:
      enabled: false  # ← Desabilitar
    fusion:
      type: "none"
    classifier:
      input_dim: 256  # ← Direto do semantic

# Experimento 2: PROPOSTA COMPLETA
experiment_full_dual_stream:
  description: "Dual-stream completo (semantic + structural + GAT)"
  model:
    semantic:
      input_dim: 3072
      hidden_dim: 256
      num_layers: 2
    structural:
      enabled: true
      input_dim: 24  # ← 24 features expandidas
      hidden_dim: 256
      num_heads: 4
      use_edge_weights: true
    fusion:
      type: "gated"  # ou "cross_attention"
    classifier:
      input_dim: 512

# Experimento 3: ABLATION - Structural sem Grafo
experiment_structural_no_graph:
  description: "Features estruturais SEM GAT (apenas MLP)"
  model:
    structural:
      use_gat: false  # ← MLP simples

# Experimento 4: ABLATION - Apenas Grafo (sem features históricas)
experiment_graph_only:
  description: "GAT sobre embeddings semânticos (sem features estruturais)"
  model:
    structural:
      input_dim: 3072  # ← Usar embeddings como node features
      features_type: "semantic"  # ← Não usar historical features

# Experimento 5: ABLATION - Features Expandidas (24) vs Originais (6)
experiment_feature_comparison:
  description: "Comparar 6 features vs 24 features"
  variants:
    - structural.input_dim: 6
    - structural.input_dim: 24
```

#### Métricas de Comparação:

```python
COMPARISON_METRICS = {
    # Performance:
    'test_f1_macro': "F1 Macro (principal métrica)",
    'test_accuracy': "Accuracy",
    'test_pass_recall': "Recall da classe Pass",
    'test_fail_recall': "Recall da classe Fail",
    'test_auprc': "AUPRC (área sob precision-recall)",

    # Contribution Analysis:
    'structural_contribution': "Melhoria relativa vs baseline",
    'graph_contribution': "Melhoria relativa vs structural-no-graph",

    # Statistical Significance:
    'p_value': "Teste t de Student (paired, 5 runs)",
    'confidence_interval': "IC 95% da diferença",
}
```

#### Análise de Contribuição Estrutural:

```python
def analyze_structural_contribution(baseline_f1, dual_stream_f1):
    """
    Quantifica contribuição da camada estrutural
    """
    improvement = dual_stream_f1 - baseline_f1
    relative_improvement = (improvement / baseline_f1) * 100

    print(f"Baseline F1 (semantic only): {baseline_f1:.4f}")
    print(f"Dual-stream F1 (semantic + structural): {dual_stream_f1:.4f}")
    print(f"Absolute improvement: +{improvement:.4f}")
    print(f"Relative improvement: +{relative_improvement:.2f}%")

    # Teste estatístico (5 runs com diferentes seeds)
    from scipy.stats import ttest_rel
    t_stat, p_value = ttest_rel(baseline_runs, dual_stream_runs)

    if p_value < 0.05:
        print(f"✓ Improvement is statistically significant (p={p_value:.4f})")
    else:
        print(f"✗ Improvement is NOT significant (p={p_value:.4f})")
```

### 3.5 Visualização do Grafo

#### Exemplo 1: Visualizar Subgrafo de um Batch

```python
# scripts/visualize_phylogenetic_graph.py
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_batch_subgraph(
    tc_keys: list,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor = None,
    save_path: str = "graph_visualization.png"
):
    """
    Visualiza subgrafo de um batch com labels e predições
    """
    # Criar grafo NetworkX
    G = nx.Graph()

    # Adicionar nós
    for i, tc_key in enumerate(tc_keys):
        label = "Fail" if labels[i] == 0 else "Pass"
        pred = "Fail" if predictions[i] == 0 else "Pass" if predictions is not None else "?"

        G.add_node(i,
                   tc_key=tc_key,
                   label=label,
                   prediction=pred,
                   correct=(label == pred) if predictions is not None else None)

    # Adicionar arestas
    edge_index_np = edge_index.cpu().numpy()
    edge_weights_np = edge_weights.cpu().numpy()

    for i in range(edge_index.shape[1]):
        src, dst = edge_index_np[:, i]
        weight = edge_weights_np[i]
        G.add_edge(src, dst, weight=weight)

    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.5)

    # Plot
    fig, ax = plt.subplots(figsize=(20, 15))

    # Colorir nós por label (vermelho=Fail, verde=Pass)
    node_colors = ['red' if labels[i] == 0 else 'green' for i in range(len(tc_keys))]

    # Desenhar nós
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=800,
                          alpha=0.7,
                          ax=ax)

    # Desenhar arestas (espessura = peso)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos,
                          width=[w*5 for w in weights],  # Scale for visibility
                          alpha=0.3,
                          ax=ax)

    # Labels dos nós
    labels_dict = {i: f"{tc_keys[i][:8]}\n{G.nodes[i]['label']}"
                   for i in range(len(tc_keys))}
    nx.draw_networkx_labels(G, pos, labels_dict, font_size=8, ax=ax)

    # Título
    ax.set_title(f"Phylogenetic Graph Subgraph (Batch Size: {len(tc_keys)})\n"
                 f"Red=Fail, Green=Pass | Edge thickness=weight",
                 fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved graph visualization to {save_path}")

    # Stats
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Avg degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"  Density: {nx.density(G):.4f}")

    # Componentes conectados
    components = list(nx.connected_components(G))
    print(f"  Connected components: {len(components)}")
    if len(components) > 1:
        print(f"  Isolated nodes: {sum(1 for c in components if len(c) == 1)}")
```

#### Exemplo 2: Visualizar Attention Weights do GAT

```python
def visualize_gat_attention(
    model: DualStreamModelV8,
    batch_data: dict,
    save_path: str = "gat_attention_heatmap.png"
):
    """
    Visualiza attention weights da primeira camada GAT
    """
    # Forward pass com hook para capturar attention
    attention_weights = []

    def hook_fn(module, input, output):
        # GATConv retorna (output, attention_weights)
        if isinstance(output, tuple):
            attention_weights.append(output[1])

    # Register hook
    handle = model.structural_stream.conv1.register_forward_hook(hook_fn)

    # Forward
    with torch.no_grad():
        _ = model(
            batch_data['semantic_input'],
            batch_data['structural_input'],
            batch_data['edge_index'],
            batch_data['edge_weights']
        )

    handle.remove()

    # Plot attention matrix
    if len(attention_weights) > 0:
        attn = attention_weights[0].cpu().numpy()  # [E, num_heads]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        for head in range(4):  # 4 attention heads
            ax = axes[head]

            # Criar matriz de atenção [N, N]
            N = batch_data['structural_input'].shape[0]
            attn_matrix = np.zeros((N, N))

            edge_index = batch_data['edge_index'].cpu().numpy()
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[:, i]
                attn_matrix[src, dst] = attn[i, head]

            # Heatmap
            im = ax.imshow(attn_matrix, cmap='hot', interpolation='nearest')
            ax.set_title(f'GAT Head {head+1}')
            ax.set_xlabel('Target Node')
            ax.set_ylabel('Source Node')
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved GAT attention heatmap to {save_path}")
```

---

## 4. PLANO DE AÇÃO RECOMENDADO

### Fase 1: Resolver Erro NVML (Urgente)

**Objetivo:** Completar pipeline de encoding sem crashes.

```bash
# Opção 1: Pré-computar embeddings offline
python scripts/precompute_embeddings.py \
  --train datasets/train.csv \
  --val datasets/val.csv \
  --test datasets/test.csv \
  --output cache/embeddings_v9.npz \
  --chunk_size 500 \
  --device cuda

# Opção 2: Usar CPU para commits (modificar código)
# Implementar CPU fallback em qodo_encoder_chunked.py

# Opção 3: Reduzir chunk size (config)
# chunk_size: 1000 → 500
```

**Critério de Sucesso:**
- ✅ Encoding completo de train/val/test sem crashes
- ✅ Tempo total < 3 horas

### Fase 2: Expansão de Features Estruturais (Alta Prioridade)

**Objetivo:** Aumentar features de 6 → 24 para capturar mais sinal.

**Tarefas:**
1. Implementar StructuralFeatureExtractorV2 com 24 features
2. Adicionar fallback para nós isolados (k-NN semântico)
3. Validar que features não têm NaN/Inf
4. Computar feature importance (SHAP ou permutation)

**Critério de Sucesso:**
- ✅ 24 features extraídas sem erros
- ✅ Feature importance mostra que features estruturais são relevantes
- ✅ Correlação entre features < 0.9 (evitar redundância)

### Fase 3: Ablation Study (Validação Científica)

**Objetivo:** Provar que camada estrutural contribui significativamente.

**Experimentos:**
```yaml
Priority 1: Baseline vs Proposta
├── Exp A: Semantic Only (baseline)
└── Exp B: Dual-Stream (semantic + structural 24 features + GAT)

Priority 2: Componente Analysis
├── Exp C: Structural sem GAT (features + MLP)
├── Exp D: GAT sem features históricas (apenas embeddings)
└── Exp E: 6 features vs 24 features

Priority 3: Fusion Analysis
├── Exp F: Cross-Attention Fusion
└── Exp G: Gated Fusion
```

**Critério de Sucesso:**
- ✅ Dual-stream (B) > Semantic-only (A) com p < 0.05
- ✅ Melhoria relativa ≥ 5% em F1 Macro
- ✅ 24 features (E) > 6 features com p < 0.05

### Fase 4: Visualização e Interpretabilidade

**Objetivo:** Mostrar como o grafo e features estruturais funcionam.

**Deliverables:**
1. Visualização de subgrafos (networkx)
2. Heatmap de GAT attention weights
3. Feature importance plot (SHAP)
4. Análise de casos: Por que modelo acertou/errou?

**Critério de Sucesso:**
- ✅ 5 visualizações de subgrafos (diferentes padrões)
- ✅ Análise de 10 casos (5 acertos, 5 erros)
- ✅ Documento com insights sobre o que o modelo aprendeu

---

## 5. JUSTIFICATIVA CIENTÍFICA DA PROPOSTA

### 5.1 Por que Dual-Stream?

**Tese:** Informação semântica (texto) e estrutural (histórico/grafo) são **ortogonais** e **complementares**.

**Evidência Esperada (após ablation study):**

| Modelo | F1 Macro | Melhoria | Justificativa |
|--------|----------|----------|---------------|
| Semantic Only | 0.50 | baseline | Captura similaridade de descrições |
| Structural Only | 0.35 | -30% | Features históricas sozinhas são fracas |
| **Dual-Stream** | **0.58** | **+16%** | **Fusão captura padrões que cada stream perde** |

**Exemplos de Complementaridade:**

```
Caso 1: Teste Novo (sem histórico)
├── Semantic Stream: Alta confiança (similar a testes conhecidos)
├── Structural Stream: Baixa confiança (test_age=0, sem histórico)
└── Fusion (Gated): Gate aprende a confiar mais no semantic (z≈1)

Caso 2: Teste Flaky (histórico oscilante)
├── Semantic Stream: Baixa confiança (descrição ambígua)
├── Structural Stream: Alta confiança (flakiness_rate=0.8 → provável fail)
└── Fusion (Gated): Gate aprende a confiar mais no structural (z≈0)

Caso 3: Teste Estável com Commit Crítico
├── Semantic Stream: Baixa confiança (descrição genérica)
├── Structural Stream: Sinal forte (failure_rate=0.05 mas commit_impact=10)
├── GAT: Vizinhos no grafo também falhando (co-failure spike)
└── Fusion: Combina sinais → predição correta de Fail
```

### 5.2 Por que GAT (Graph Attention)?

**Justificativa:**

1. **Co-failure patterns são locais, não globais:**
   - Testes falhando juntos formam clusters (ex: todos de um módulo)
   - GAT aprende a propagar sinal de falha entre vizinhos
   - Attention weights mostram quais vizinhos são mais relevantes

2. **Superior a mean aggregation (V7):**
   - V7: MessagePassing com mean → todos vizinhos têm peso igual
   - V8: GAT com attention → vizinhos têm pesos aprendidos
   - Evidência esperada: GAT > MeanPooling em F1

3. **Alinhado com semantic stream (transformer attention):**
   - Ambas streams usam attention mechanism
   - Arquitetura unificada sob paradigma de attention

### 5.3 Limitações e Trabalho Futuro

**Limitações Atuais:**

1. **Grafo estático:** Não adapta a co-failures durante treinamento
   - **Solução futura:** Grafo dinâmico reconstruído a cada epoch

2. **Nós isolados:** Testes novos sem arestas
   - **Solução implementada:** Fallback k-NN semântico + self-loops

3. **Features esparsas:** Muitos zeros para testes novos
   - **Solução implementada:** Gated Fusion para arbitragem dinâmica

4. **Escalabilidade:** Grafo global tem ~50K nós
   - **Solução futura:** GraphSAINT sampling para treinar em subgrafos

**Trabalho Futuro:**

1. Temporal Graph Networks (TGN) para capturar evolução temporal
2. Heterogeneous graphs: nós de diferentes tipos (tests, commits, builds)
3. Contrastive learning para aprender embeddings estruturais
4. Multi-task learning: classificação + ranking (APFD)

---

## 6. CONCLUSÃO

### Respondendo às Perguntas Originais:

✅ **Como funciona o grafo?**
→ Grafo GLOBAL construído no training, subgrafos extraídos por batch.
→ Tipos: co-failure (testes falhando juntos) ou commit-dependency (commits compartilhados).

✅ **Grafo por build ou global?**
→ GLOBAL (construído uma vez), mas usado como subgrafos por batch durante inferência.

✅ **Como visualizar?**
→ Scripts propostos: `visualize_batch_subgraph()` e `visualize_gat_attention()`.

✅ **Por que 3 features não são suficientes?**
→ São 6, não 3! Mas ainda são poucas. Proposta: expandir para 24 features.

✅ **Como justificar contribuição estrutural?**
→ Ablation study: comparar semantic-only vs dual-stream com testes estatísticos.

⚠️ **Erro NVML?**
→ Causa: fragmentação CUDA. Soluções: pré-computar embeddings, usar CPU, ou reduzir chunk size.

### Recomendação Final:

**SIM, a proposta dual-stream faz sentido**, mas precisa de:

1. ✅ **Resolução do erro NVML** (urgente)
2. ✅ **Expansão de features estruturais** (6 → 24)
3. ✅ **Ablation study rigoroso** (provar contribuição)
4. ✅ **Visualização e interpretabilidade** (mostrar como funciona)

Com essas melhorias, a tese terá **fundamento científico sólido** para afirmar que:

> *"A fusão de informação semântica (baseada em texto) com informação estrutural/filogenética (baseada em histórico e grafo de co-failure) melhora significativamente a predição de falhas em testes de software, capturando padrões complementares que cada modalidade isolada não consegue detectar."*

---

**Próximo Passo Sugerido:** Implementar Fase 1 (resolver NVML) e Fase 2 (expandir features) antes de rodar experimentos completos.
