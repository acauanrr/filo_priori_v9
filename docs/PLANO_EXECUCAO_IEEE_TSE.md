# PLANO DE EXECUCAO - Filo-Priori IEEE TSE

**Documento**: Plano de Execucao para Publicacao no IEEE Transactions on Software Engineering
**Versao**: 1.0
**Data**: Novembro 2025
**Status**: Em Execucao

---

## 1. SUMARIO EXECUTIVO

### 1.1 Objetivo
Transformar o Filo-Priori de uma abordagem generica de ML para TCP em uma **abordagem filogenetica inovadora** baseada em GNNs, adequada para publicacao no IEEE TSE.

### 1.2 Mudanca de Paradigma
| Aspecto | Abordagem Anterior | Nova Abordagem Filogenetica |
|---------|-------------------|----------------------------|
| Historico | Serie temporal linear | Arvore filogenetica (Git DAG) |
| Metafora | Machine Learning generico | Bio-inspirada (Evolucao de Software) |
| Grafo | Multi-edge estatico | Grafo Temporal Evolutivo |
| Atencao | GAT simples | Atencao Hierarquica (Micro/Meso/Macro) |
| Diferencial | Ranking-aware loss | Distancia Filogenetica + Sinal Evolutivo |

### 1.3 Score Alvo
| Metrica | Atual | Meta | Gap |
|---------|-------|------|-----|
| Novidade | 7/10 | 9/10 | +2 |
| Rigor Metodologico | 8/10 | 9/10 | +1 |
| Fundamentacao Teorica | 6/10 | 9/10 | +3 |
| Referencias | 8/10 | 9/10 | +1 |
| **TOTAL** | **7.8/10** | **9.0/10** | +1.2 |

---

## 2. RESULTADOS DA RSL

### 2.1 String de Busca Executada
```
("test case prioritization" OR "regression testing" OR "test suite reduction"
OR "fault localization")
AND
("graph neural network" OR "GNN" OR "graph convolutional network" OR "GCN"
OR "graph attention network" OR "GAT" OR "attention mechanism" OR "transformer"
OR "deep learning" OR "representation learning")
AND
("phylogen*" OR "evolutionary history" OR "software evolution" OR "version control"
OR "commit graph" OR "git history" OR "lineage" OR "history-aware" OR "temporal graph"
OR "tree structure")
```

### 2.2 Papers Identificados

#### IEEE Xplore (4 papers)
| ID | DOI/URL | Topico Provavel |
|----|---------|-----------------|
| IEEE-01 | 11071794 | Temporal GNN + Fault Localization |
| IEEE-02 | 11225352 | GNN + Software Testing |
| IEEE-03 | 9787970 | History-aware Testing |
| IEEE-04 | 9796454 | Evolution-based Prioritization |

#### ACM Digital Library (8 papers)
| ID | DOI | Topico Provavel |
|----|-----|-----------------|
| ACM-01 | 10.1145/3660793 | GNN + Software Engineering |
| ACM-02 | 10.1145/3540250.3549137 | Commit-based Analysis |
| ACM-03 | 10.1145/3660798 | Test Prioritization ML |
| ACM-04 | 10.1145/3664597 | Temporal Analysis |
| ACM-05 | 10.1145/3672450 | Graph-based Testing |
| ACM-06 | 10.1145/3720526 | Evolution Analysis |
| ACM-07 | 10.1145/3637528.3671933 | Deep Learning Testing |
| ACM-08 | 10.1145/3731597 | Software Evolution |

### 2.3 Papers Complementares Identificados
| Paper | Venue | Relevancia |
|-------|-------|------------|
| NodeRank | EMSE 2024 | GNN para priorizacao de testes |
| ActGraph | arXiv 2022 | Activation Graph para TCP |
| SETS | TOSEM 2024 | DNN Test Selection |
| Neural Network Embeddings TCP | arXiv 2020 | Embeddings para TCP |

---

## 3. ARQUITETURA FILO-PRIORI REVISADA

### 3.1 Visao Geral da Nova Arquitetura

```
                    FILO-PRIORI: ARQUITETURA FILOGENETICA
    =====================================================================

    ENTRADA                           PROCESSAMENTO                    SAIDA
    -------                           ------------                     -----

    [Git DAG]                    +------------------+
    Commits +                    |  PHYLO-ENCODER   |
    Branches +  ---------------→ |  (GGNN Temporal) | ----+
    Merges                       |  Distancia Filog.|     |
                                 +------------------+     |
                                                          |     +-------------+
    [Codigo Fonte]               +------------------+     +--→  | CROSS-      |
    AST/CFG +                    |  CODE-ENCODER    |     |     | ATTENTION   |
    Diffs +     ---------------→ |  (GAT + CodeBERT)| ----+     | FUSION      |
    Cobertura                    |  Semantica Prof. |           |             |
                                 +------------------+           +------+------+
                                                                       |
    [Resultados Tests]           +------------------+                  v
    Pass/Fail +                  |  HISTORY-ENCODER |           +-------------+
    Flakiness + ---------------→ |  (Attention      | ----+     | RANKING     |
    Tempo Exec                   |   Hierarquica)   |     |     | MODULE      |
                                 +------------------+     +--→  | P(falha|t)  |
                                                                +-------------+
                                                                       |
                                                                       v
                                                                [Lista Priorizada]
                                                                T' = {t1, t2, ..., tn}
```

### 3.2 Modulo 1: Phylo-Encoder (Codificador Filogenetico)

**Objetivo**: Aprender representacoes do contexto evolutivo respeitando a topologia do Git DAG.

**Entrada**:
- Subgrafo do historico: commit atual C_curr e k ancestrais
- Atributos por no: autor, mensagem (via BERT), R_tests (resultados)

**Arquitetura**:
```python
class PhyloEncoder(nn.Module):
    """
    Gated Graph Neural Network (GGNN) temporal para commits.
    Propaga informacao dos ancestrais para descendentes.
    """
    def __init__(self, hidden_dim=256, num_layers=3):
        self.ggnn = GGNN(hidden_dim, num_layers)
        self.distance_kernel = PhylogeneticDistanceKernel()
        self.commit_encoder = BERTEncoder('microsoft/codebert-base')

    def forward(self, commit_graph):
        # Codifica mensagens de commit
        node_features = self.commit_encoder(commit_graph.messages)

        # Calcula distancia filogenetica
        phylo_distances = self.distance_kernel(commit_graph)

        # Propaga com atenuacao por distancia
        h = self.ggnn(node_features, commit_graph.edges, phylo_distances)

        return h
```

**Kernel de Distancia Filogenetica**:
```
d_phylo(c_i, c_j) = shortest_path(c_i, c_j) * decay_factor^(num_merges)
```

### 3.3 Modulo 2: Code-Encoder (Codificador Estrutural)

**Objetivo**: Entender a semantica das mudancas de codigo.

**Entrada**:
- Grafo heterogeneo: nos de Metodos + nos de Testes
- Arestas: relacao "cobre" (cobertura dinamica)

**Arquitetura**:
```python
class CodeEncoder(nn.Module):
    """
    GAT + CodeBERT para embeddings semanticos de codigo.
    """
    def __init__(self, hidden_dim=128, num_heads=4):
        self.codebert = CodeBERTEncoder()
        self.gat = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads)

    def forward(self, code_graph):
        # Embeddings semanticos via CodeBERT
        method_embeds = self.codebert(code_graph.methods)
        test_embeds = self.codebert(code_graph.tests)

        # Atencao sobre grafo de cobertura
        h = self.gat(torch.cat([method_embeds, test_embeds]),
                     code_graph.coverage_edges)

        return h
```

### 3.4 Modulo 3: Atencao Hierarquica

**Tres Niveis de Atencao**:

1. **Nivel Micro (Codigo)**: Quais tokens/instrucoes sao suspeitos?
2. **Nivel Meso (Grafo de Chamadas)**: Quais metodos impactam quais testes?
3. **Nivel Macro (Historico)**: Quais ancestrais sao relevantes?

```python
class HierarchicalAttention(nn.Module):
    """
    Atencao em tres niveis: Micro, Meso, Macro.
    """
    def __init__(self, hidden_dim=256):
        self.micro_attention = MultiHeadAttention(hidden_dim, num_heads=8)
        self.meso_attention = GATv2Conv(hidden_dim, hidden_dim, heads=4)
        self.macro_attention = TemporalAttention(hidden_dim)

    def forward(self, phylo_embed, code_embed, history_embed):
        # Nivel Micro: atencao em tokens
        micro_out = self.micro_attention(code_embed)

        # Nivel Meso: atencao no grafo de chamadas
        meso_out = self.meso_attention(micro_out)

        # Nivel Macro: atencao temporal/filogenetica
        macro_out = self.macro_attention(phylo_embed, history_embed)

        # Fusao Cross-Attention
        fused = self.cross_attention(meso_out, macro_out)

        return fused
```

### 3.5 Modulo de Ranking

**Funcao de Pontuacao**:
```
P(falha|t) = sigmoid(W_f * Concat(h_phylo, h_struct, h_attention))
```

**Loss Combinada**:
```
L = lambda_1 * L_focal + lambda_2 * L_rank + lambda_3 * L_phylo_reg
```

Onde `L_phylo_reg` e um termo de regularizacao filogenetica que penaliza predicoes inconsistentes com a estrutura evolutiva.

---

## 4. CRONOGRAMA DE EXECUCAO

### Fase 1: Fundamentacao (Semana 1-2)
| Tarefa | Responsavel | Status | Entrega |
|--------|-------------|--------|---------|
| Extrair dados dos 12 papers RSL | Pesquisador | Pendente | Sem 1 |
| Atualizar referencias.bib | Pesquisador | Pendente | Sem 1 |
| Reescrever Background | Pesquisador | Pendente | Sem 2 |
| Reescrever Related Work | Pesquisador | Pendente | Sem 2 |

### Fase 2: Metodologia (Semana 3-4)
| Tarefa | Responsavel | Status | Entrega |
|--------|-------------|--------|---------|
| Implementar PhyloEncoder | Desenvolvedor | Pendente | Sem 3 |
| Implementar HierarchicalAttention | Desenvolvedor | Pendente | Sem 3 |
| Integrar com pipeline existente | Desenvolvedor | Pendente | Sem 4 |
| Escrever secao Approach | Pesquisador | Pendente | Sem 4 |

### Fase 3: Experimentos (Semana 5-6)
| Tarefa | Responsavel | Status | Entrega |
|--------|-------------|--------|---------|
| Executar experimentos comparativos | Desenvolvedor | Pendente | Sem 5 |
| Calcular metricas (APFD, APFD-c) | Desenvolvedor | Pendente | Sem 5 |
| Analise estatistica (Wilcoxon) | Pesquisador | Pendente | Sem 6 |
| Atualizar secao Results | Pesquisador | Pendente | Sem 6 |

### Fase 4: Finalizacao (Semana 7-8)
| Tarefa | Responsavel | Status | Entrega |
|--------|-------------|--------|---------|
| Revisao completa do paper | Equipe | Pendente | Sem 7 |
| English proofreading | Externo | Pendente | Sem 7 |
| Preparar replication package | Desenvolvedor | Pendente | Sem 8 |
| Submissao IEEE TSE | Equipe | Pendente | Sem 8 |

---

## 5. MAPEAMENTO CONCEITUAL

### 5.1 Biologia vs Engenharia de Software

| Conceito Biologico | Equivalente em ES | Aplicacao Filo-Priori |
|-------------------|-------------------|----------------------|
| Taxon/Especie | Versao/Commit | No no Grafo Filogenetico |
| Sequencia DNA | Codigo Fonte/AST | Entrada para Code-Encoder |
| Mutacao (SNP) | Diff de Codigo | Arestas ponderadas |
| Arvore Filogenetica | Git DAG | Estrutura topologica GGNN |
| Sinal Filogenetico | Autocorrelacao de Falhas | Peso aprendido por atencao |
| Ancestral Comum | Merge Base | Ponto de sincronizacao |

### 5.2 Gap Enderecado

| Gap | Tecnicas Anteriores | Solucao Filo-Priori |
|-----|---------------------|---------------------|
| Independencia | Assume observacoes independentes | Modela dependencia via grafo |
| Semantica | Features manuais | GNN + CodeBERT |
| Topologia | Serie temporal linear | Git DAG como estrutura nativa |
| Interpretabilidade | Caixa-preta | Atencao explicavel |

---

## 6. ESTRUTURA DO PAPER IEEE TSE

### 6.1 Outline Revisado

```
1. Introduction (3 paginas)
   1.1 O Problema da Explosao de Testes
   1.2 Limitacoes das Abordagens Atuais
   1.3 A Metafora Filogenetica
   1.4 Contribuicoes
   1.5 Organizacao do Paper

2. Background (3 paginas)
   2.1 Test Case Prioritization
   2.2 Metrica APFD
   2.3 Filogenética Computacional
   2.4 Graph Neural Networks
   2.5 Mecanismos de Atencao

3. Related Work (4 paginas)
   3.1 TCP Tradicional (Cobertura)
   3.2 TCP com Machine Learning
   3.3 TCP com Deep Learning
   3.4 GNN em Engenharia de Software
   3.5 Analise Evolutiva de Software
   3.6 Posicionamento do Filo-Priori

4. Approach: Filo-Priori (5 paginas)
   4.1 Visao Geral
   4.2 Construcao do Grafo Filogenetico
   4.3 Phylo-Encoder (GGNN Temporal)
   4.4 Code-Encoder (GAT + CodeBERT)
   4.5 Atencao Hierarquica
   4.6 Modulo de Ranking
   4.7 Funcao de Perda Combinada

5. Experimental Design (3 paginas)
   5.1 Research Questions
   5.2 Datasets
   5.3 Baselines
   5.4 Metricas (APFD, APFD-c, Top-k)
   5.5 Implementacao

6. Results (5 paginas)
   6.1 RQ1: Efetividade
   6.2 RQ2: Contribuicao dos Componentes
   6.3 RQ3: Robustez Temporal
   6.4 RQ4: Impacto da Distancia Filogenetica

7. Discussion (3 paginas)
   7.1 Interpretacao dos Resultados
   7.2 Implicacoes Praticas
   7.3 Limitacoes

8. Threats to Validity (2 paginas)

9. Conclusion (1 pagina)

References (50-60 referencias)
```

---

## 7. ACOES IMEDIATAS

### 7.1 Proximas 48 horas
- [ ] Ler PDFs da pasta RSL/ e extrair dados
- [ ] Atualizar references_ieee.bib com papers da RSL
- [ ] Iniciar reescrita da Introduction com foco filogenetico

### 7.2 Proxima Semana
- [ ] Completar Background e Related Work
- [ ] Implementar prototipo do PhyloEncoder
- [ ] Criar figuras de arquitetura

### 7.3 Proximo Mes
- [ ] Paper completo em draft
- [ ] Experimentos executados
- [ ] Revisao interna

---

## 8. METRICAS DE SUCESSO

| Metrica | Criterio de Sucesso |
|---------|---------------------|
| APFD | >= 0.65 (superando baselines) |
| APFD vs NodeRank | Diferenca estatisticamente significativa |
| Ablation | Phylo-Encoder contribui >= 10% |
| Temporal CV | Degradacao < 5% ao longo do tempo |
| Paper | Aceito no IEEE TSE |

---

**Documento criado**: Novembro 2025
**Proxima revisao**: Semanal
