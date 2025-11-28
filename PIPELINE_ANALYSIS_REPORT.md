# Filo-Priori V9: Relatório de Análise do Pipeline

## Objetivo
Este documento analisa o que está **realmente implementado e funcionando** no pipeline vs. o que está **descrito no paper IEEE TSE**.

---

## 1. PIPELINE REAL IMPLEMENTADO (main.py)

### 1.1 Modelo Principal: `DualStreamModelV8`
**Arquivo**: `src/models/dual_stream_v8.py`

O modelo que está sendo usado é **DualStreamModelV8** com os seguintes componentes:

| Componente | Implementação | Descrição |
|------------|---------------|-----------|
| SemanticStream | FFN com residual | Processa embeddings SBERT [batch, 1536] |
| StructuralStreamV8 | **GATConv** (PyG) | Processa features estruturais [batch, 10] com graph attention |
| Fusion | CrossAttentionFusion ou GatedFusionUnit | Fusão bidirecional |
| Classifier | SimpleClassifier (MLP) | Classificação binária Pass/Fail |

### 1.2 Features Utilizadas

**Embeddings Semânticos:**
- Modelo: `sentence-transformers/all-mpnet-base-v2` (SBERT)
- Dimensão: 768 * 2 = 1536 (TC + Commit concatenados)

**Features Estruturais (V2.5 - 10 features selecionadas):**
1. test_age
2. failure_rate
3. recent_failure_rate
4. flakiness_rate
5. commit_count
6. test_novelty
7. consecutive_failures
8. max_consecutive_failures
9. failure_trend
10. cr_count

### 1.3 Grafo
- **Tipo**: Multi-edge graph
- **Arestas**: co_failure (1.0), co_success (0.5), semantic (0.3)
- **Construção**: Baseado em co-ocorrência de falhas e similaridade semântica

### 1.4 Loss Function
**Arquivo**: `src/training/losses.py`

**Tipos disponíveis:**
- CrossEntropyLoss (ce)
- WeightedCrossEntropyLoss (weighted_ce)
- FocalLoss (focal)
- WeightedFocalLoss (weighted_focal)

**NÃO implementado no main.py:**
- Ranking Loss (RankNet)
- Phylogenetic Regularization

### 1.5 Treinamento (main.py)
```python
criterion = create_loss_function(config, class_weights_tensor)  # Apenas Focal/CE
# NÃO há ranking loss
# NÃO há phylo regularization
```

---

## 2. O QUE O PAPER DESCREVE (main_ieee_tse.tex)

### 2.1 Arquitetura Descrita

| Componente | Descrito no Paper | Status |
|------------|------------------|--------|
| Phylo-Encoder LITE (GGNN) | 2 layers, 128-dim, Git DAG | EXISTE no código mas NÃO usado |
| Code-Encoder (GATv2) | GATv2 sobre grafo de testes | Usa GATConv (não GATv2 explícito) |
| Hierarchical Attention | Micro/Meso/Macro | EXISTE no código mas NÃO usado |
| Cross-Attention Fusion | Phylo + Structural → Semantic | Apenas Semantic + Structural |
| Combined Loss | Focal + Ranking + Phylo-Reg | Apenas Focal Loss |

### 2.2 Loss Function Descrita
```
L = λ₁·L_focal + λ₂·L_rank + λ₃·L_phylo
λ₁ = 0.7, λ₂ = 0.3, λ₃ = 0.05
```

**Realidade:**
```
L = L_focal (ou L_weighted_focal)
# Sem ranking loss
# Sem phylo regularization
```

### 2.3 Seções Problemáticas no Paper

**Seção 4.4 (Phylo-Encoder LITE):**
- Descreve GGNN com temperatura aprendível
- Código existe em `phylo_encoder.py` mas NÃO é usado no pipeline principal

**Seção 4.6 (Hybrid Fusion Architecture):**
- Descreve combinação phylo + structural via addition
- Realidade: apenas semantic + structural (sem phylo)

**Seção 4.7 (Ranking-Aware Training):**
- Descreve combined loss com 3 componentes
- Realidade: apenas focal loss

---

## 3. CÓDIGO QUE EXISTE MAS NÃO É USADO

### 3.1 PhylogeneticDualStreamModel
**Arquivo**: `src/models/phylogenetic_dual_stream.py`

Este modelo inclui:
- PhyloEncoder (GGNN)
- HierarchicalAttention
- PhylogeneticRegularization

**Problema**: O `main.py` usa `create_model()` que por padrão cria `DualStreamModelV8`, não `PhylogeneticDualStreamModel`.

### 3.2 PhyloEncoder
**Arquivo**: `src/models/phylo_encoder.py`

Implementação completa de:
- PhylogeneticDistanceKernel (com temperatura aprendível)
- GGNNLayer (Gated Graph Neural Network)
- PhyloEncoder

**Problema**: Nunca instanciado no pipeline principal.

### 3.3 Configs que não são usadas
**Arquivo**: `configs/experiment_phylogenetic.yaml`

Define:
```yaml
model:
  type: "phylogenetic_dual_stream"
training:
  ranking:
    enabled: true
    weight: 0.3
```

**Problema**: O `main.py` não implementa a lógica para usar ranking loss, mesmo que a config defina.

---

## 4. RESULTADOS REPORTADOS vs IMPLEMENTAÇÃO

### 4.1 Ablation Study (results_ieee.tex)
O paper reporta:
- "w/o Ranking Loss" contribui +3.5%
- "w/o Graph Attention" contribui +17.0%

**Inconsistência**: Se ranking loss não está implementado no main.py, como foi feito o ablation?

**Possibilidades:**
1. Experimentos foram feitos com código diferente (não commitado)
2. Resultados são de versão anterior
3. "Ranking Loss" refere-se a algo diferente

### 4.2 APFD Reportado
- Paper: APFD = 0.6413
- Baseline FailureRate: APFD = 0.6289
- Melhoria: +2.0%

**Conclusão**: Os resultados provavelmente vêm do modelo simples (DualStreamModelV8 com Focal Loss), não da arquitetura filogenética completa.

---

## 5. RECOMENDAÇÕES PARA ATUALIZAÇÃO DO PAPER

### Opção A: Atualizar Paper para Refletir Implementação Real
1. Remover descrições de Phylo-Encoder e Hierarchical Attention
2. Remover descrição de Ranking Loss e Phylo Regularization
3. Descrever arquitetura como: SBERT + GAT + CrossAttention + Focal Loss
4. Manter resultados (foram obtidos com esta config)

### Opção B: Implementar o que está Descrito
1. Modificar main.py para usar PhylogeneticDualStreamModel
2. Implementar combined loss (Focal + Ranking + Phylo-Reg)
3. Re-executar experimentos
4. Atualizar resultados

### Recomendação
**Opção A** é mais segura:
- Resultados já validados
- Arquitetura funcional
- Evita re-executar todos os experimentos

---

## 6. MAPEAMENTO PAPER → CÓDIGO

| Seção Paper | Componente Descrito | Arquivo Código | Status |
|-------------|---------------------|----------------|--------|
| 4.2 | SBERT Encoding | `src/embeddings/sbert_encoder.py` | USADO |
| 4.3 | Structural Features | `src/preprocessing/structural_feature_extractor_v2_5.py` | USADO |
| 4.4 | Phylo-Encoder (GGNN) | `src/models/phylo_encoder.py` | EXISTE, NÃO USADO |
| 4.5 | Code-Encoder (GATv2) | `src/models/dual_stream_v8.py:StructuralStreamV8` | USADO (GATConv) |
| 4.6 | Hierarchical Attention | `src/models/phylogenetic_dual_stream.py` | EXISTE, NÃO USADO |
| 4.6 | Cross-Attention Fusion | `src/models/dual_stream_v8.py:CrossAttentionFusion` | USADO |
| 4.7 | Focal Loss | `src/training/losses.py` | USADO |
| 4.7 | Ranking Loss | - | NÃO IMPLEMENTADO |
| 4.7 | Phylo Regularization | `src/models/phylogenetic_dual_stream.py` | EXISTE, NÃO USADO |

---

## 7. RESUMO EXECUTIVO

### O que está funcionando e produzindo resultados:
1. **DualStreamModelV8** - Modelo dual-stream com GATConv
2. **SBERT Embeddings** - all-mpnet-base-v2
3. **10 Features Estruturais** - V2.5 extractor
4. **Multi-edge Graph** - co_failure + co_success + semantic
5. **Weighted Focal Loss** - Para class imbalance
6. **Cross-Attention Fusion** - Semantic + Structural

### O que está descrito mas NÃO usado:
1. Phylo-Encoder (GGNN)
2. Hierarchical Attention (Micro/Meso/Macro)
3. Ranking Loss (RankNet)
4. Phylogenetic Regularization
5. Combined Loss (3 componentes)

### Ação Necessária:
**Atualizar paper para refletir a implementação real, removendo descrições de componentes não utilizados.**
