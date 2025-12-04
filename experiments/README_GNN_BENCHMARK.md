# GNN Benchmark Experiment: Filo-Priori vs NodeRank

Este documento descreve como executar o experimento de comparação entre Filo-Priori e os métodos do paper NodeRank (IEEE TSE 2024).

## Referência

**Paper:** "Test Input Prioritization for Graph Neural Networks" (IEEE TSE 2024)

**Objetivo:** Comparar o desempenho do Filo-Priori com os métodos de priorização de testes para GNNs descritos no paper.

## Datasets

| Dataset   | Nós    | Arestas | Classes | Tipo              |
|-----------|--------|---------|---------|-------------------|
| Cora      | 2,708  | 5,429   | 7       | Rede de citações  |
| CiteSeer  | 3,327  | 4,732   | 6       | Rede de citações  |
| PubMed    | 19,717 | 44,338  | 3       | Rede de citações  |

> **Nota:** O dataset LastFM Asia não está disponível devido a erro 404 no servidor original.

## Arquitetura do Filo-Priori para GNN

O modelo foi adaptado da versão que alcançou **APFD 0.7595** no dataset industrial:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      GNN FILO-PRIORI ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Node Features [N, F]                                               │
│        │                                                            │
│        ├──► NodeFeatureStream (FFN) ──► [N, 256]                   │
│        │                                    │                       │
│        └──► StructuralStreamGAT ───────► [N, 256]                  │
│                    │                        │                       │
│                    │    ┌───────────────────┘                       │
│                    │    │                                           │
│                    ▼    ▼                                           │
│              CrossAttentionFusion                                   │
│                    │                                                │
│                    ▼                                                │
│               Classifier (MLP)                                      │
│                    │                                                │
│                    ▼                                                │
│              Logits [N, C]                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Componentes Principais

1. **NodeFeatureStream**: FFN com residual connections (substitui SBERT encoder)
2. **StructuralStreamGAT**: 2-layer GAT com multi-head attention (usa o grafo nativo)
3. **CrossAttentionFusion**: Fusão bidirecional com atenção cruzada
4. **Focal Loss**: Para lidar com desbalanceamento de classes
5. **Uncertainty Features**: Extraídas das predições para priorização

## Métodos de Priorização Implementados

| Método | Descrição | Fórmula |
|--------|-----------|---------|
| **Random** | Baseline aleatório | - |
| **DeepGini** | Impureza de Gini | `1 - Σ(p_i²)` |
| **Entropy** | Entropia normalizada | `-Σ(p_i × log(p_i))` |
| **VanillaSM** | Margem de softmax | `p_1 - p_2` |
| **PCS** | Least Confidence | `1 - max(p)` |
| **Filo-Priori** | Combinação multi-sinal | `α×incerteza + β×estrutura + γ×desacordo` |

## Métricas

- **APFD** - Average Percentage of Fault Detection (principal)
- **PFD@10** - % de falhas detectadas nos primeiros 10%
- **PFD@20** - % de falhas detectadas nos primeiros 20%

---

## Execução do Experimento

### Pré-requisitos

```bash
# 1. Ativar ambiente virtual
source venv/bin/activate

# 2. Verificar dependências
pip install torch torch-geometric scipy scikit-learn pandas tqdm
```

### Opção 1: Experimento Completo com Treinamento (RECOMENDADO)

Este é o experimento com o **pipeline completo do Filo-Priori**, incluindo:
- Treinamento do modelo GNN Filo-Priori
- Focal Loss para balanceamento
- Early stopping
- Extração de features de incerteza
- Cálculo de APFD/PFD

```bash
cd /home/acauan/ufam/iats/sprint_07/filo_priori_v9

# Execução padrão (5 runs por dataset)
python experiments/run_gnn_filo_priori.py

# Execução rápida (1 run)
python experiments/run_gnn_filo_priori.py --n_runs 1

# Datasets específicos
python experiments/run_gnn_filo_priori.py --datasets cora citeseer

# Ajustar hiperparâmetros
python experiments/run_gnn_filo_priori.py --num_epochs 300 --hidden_dim 128 --lr 0.001
```

**Parâmetros disponíveis:**
- `--datasets`: Lista de datasets (default: cora citeseer pubmed)
- `--n_runs`: Número de execuções por dataset (default: 5)
- `--num_epochs`: Épocas de treinamento (default: 200)
- `--hidden_dim`: Dimensão oculta do modelo (default: 256)
- `--lr`: Learning rate (default: 1e-3)
- `--output_dir`: Diretório para salvar resultados

**Resultados salvos em:**
```
experiments/results/gnn_filo_priori/
├── config.json           # Configuração usada
├── full_results_*.csv    # Resultados detalhados
└── summary_*.csv         # Estatísticas resumidas
```

### Opção 2: Experimento Simplificado (Sem Treinamento do Filo-Priori)

Este script usa modelos GNN padrão e aplica métodos de priorização heurísticos:

```bash
python experiments/run_gnn_benchmark.py

# Com opções
python experiments/run_gnn_benchmark.py --datasets cora --models gcn gat --n_runs 3
```

**Nota:** Esta versão **não treina** o modelo Filo-Priori completo. Use a Opção 1 para resultados competitivos.

---

## Estrutura do Experimento (Opção 1)

```
Para cada dataset (Cora, CiteSeer, PubMed):
    Para cada run (1 a n_runs):
        1. Carregar dataset via PyTorch Geometric
        2. Criar modelo GNNFiloPriori
           ├── NodeFeatureStream (FFN)
           ├── StructuralStreamGAT (2-layer GAT)
           ├── CrossAttentionFusion
           └── Classifier
        3. Treinar com Focal Loss + AdamW + Cosine Annealing
        4. Early stopping baseado em val_accuracy
        5. Avaliar no conjunto de teste
        6. Identificar misclassificações
        7. Para cada método de priorização:
           ├── Gerar ranking
           ├── Calcular APFD
           └── Calcular PFD@10, PFD@20
        8. Salvar resultados
```

---

## Interpretação dos Resultados

| Métrica | Valor Bom | Significado |
|---------|-----------|-------------|
| **APFD > 0.7** | Excelente | Detecta falhas muito cedo |
| **APFD ~ 0.6** | Bom | Melhor que aleatório |
| **APFD ~ 0.5** | Neutro | Equivalente a aleatório |
| **APFD < 0.5** | Ruim | Pior que aleatório |

### Comparação com NodeRank Paper

O paper NodeRank reporta (aproximadamente):

| Método     | Cora  | CiteSeer | PubMed |
|------------|-------|----------|--------|
| Random     | ~0.50 | ~0.50    | ~0.50  |
| DeepGini   | ~0.65 | ~0.62    | ~0.58  |
| GraphPrior | ~0.70 | ~0.68    | ~0.63  |
| NodeRank   | ~0.78 | ~0.75    | ~0.71  |

**Expectativa para Filo-Priori:** Competitivo com GraphPrior/NodeRank devido à arquitetura dual-stream com GAT.

---

## Arquivos do Experimento

```
filo_priori_v9/
├── src/
│   ├── models/
│   │   └── gnn_filo_priori.py        # Modelo GNN Filo-Priori
│   └── evaluation/
│       └── gnn_uncertainty_features.py # Extrator de features
├── datasets/
│   └── 03_gnn_benchmarks/
│       ├── download_datasets.py       # Download dos datasets
│       └── raw/                       # Dados brutos
└── experiments/
    ├── run_gnn_filo_priori.py         # Experimento COMPLETO
    ├── run_gnn_benchmark.py           # Experimento simplificado
    ├── README_GNN_BENCHMARK.md        # Este arquivo
    └── results/
        └── gnn_filo_priori/           # Resultados
```

---

## Troubleshooting

### Erro: "torch_geometric not installed"
```bash
pip install torch-geometric
```

### Erro: "CUDA out of memory"
```bash
export CUDA_VISIBLE_DEVICES=""
python experiments/run_gnn_filo_priori.py
```

### Erro: Dataset não encontrado
```bash
python datasets/03_gnn_benchmarks/download_datasets.py
```

### Treinamento muito lento
```bash
# Reduzir épocas e hidden_dim
python experiments/run_gnn_filo_priori.py --num_epochs 100 --hidden_dim 128
```

---

## Configuração Padrão

```python
config = {
    'seed': 42,
    'num_epochs': 200,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'patience': 20,
    'focal_alpha': 0.5,
    'focal_gamma': 2.0,

    # Pesos do Filo-Priori para combinação de sinais
    'filo_alpha': 0.4,   # Peso da incerteza
    'filo_beta': 0.3,    # Peso estrutural (degree)
    'filo_gamma': 0.3,   # Peso do desacordo com vizinhos

    # Arquitetura do modelo
    'model': {
        'hidden_dim': 256,
        'num_gat_heads': 4,
        'num_gat_layers': 2,
        'num_ffn_layers': 2,
        'fusion_type': 'cross_attention',
        'dropout': 0.3,
        'classifier_hidden_dims': [128, 64]
    }
}
```

---

*Relatório gerado em Dezembro 2025*
