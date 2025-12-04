# Experimentos - NodeRank vs Filo-Priori

Este diretório contém scripts para executar experimentos comparativos entre NodeRank e Filo-Priori no dataset Industry.

## Estrutura

```
experiments/
├── README.md                          # Este arquivo
├── run_noderank_industry.py           # Experimento NodeRank
├── run_deeporder_industry.py          # Experimento DeepOrder
└── compare_noderank_filopriori.py     # Comparação estatística
```

## Execução Manual do NodeRank

### Passo 1: Executar o experimento NodeRank

```bash
# Ativar ambiente virtual
source venv/bin/activate

# Executar experimento
python experiments/run_noderank_industry.py
```

**Tempo estimado:** 5-15 minutos (dependendo do hardware)

**Saída:**
- `results/noderank_industry/apfd_per_build_FULL_testcsv.csv` - APFD por build
- `results/noderank_industry/experiment_summary.json` - Resumo do experimento
- `results/noderank_industry/comparison_summary.txt` - Resumo legível

## Execução Manual do DeepOrder

```bash
# Ativar ambiente virtual
source venv/bin/activate

# Executar experimento
python experiments/run_deeporder_industry.py
```

**Saída:**
- `results/deeporder_industry/apfd_per_build_FULL_testcsv.csv` - APFD por build (mesmo formato do Filo-Priori)
- `results/deeporder_industry/experiment_summary.json` - Resumo do experimento
- `results/deeporder_industry/comparison_summary.txt` - Resumo legível

### Passo 2: Comparar com Filo-Priori

```bash
python experiments/compare_noderank_filopriori.py
```

**Saída:**
- `results/noderank_industry/comparison_statistics.json` - Estatísticas
- `results/noderank_industry/per_build_comparison.csv` - Comparação por build

## Garantias de Comparabilidade Científica

Os experimentos são projetados para serem cientificamente comparáveis:

1. **Mesmo Split de Dados:**
   - Treino: `datasets/01_industry/train.csv`
   - Teste: `datasets/01_industry/test.csv`

2. **Mesma Métrica (APFD):**
   - Fórmula: `APFD = 1 - Σ(rank_falhas) / (n_falhas × n_testes) + 1 / (2 × n_testes)`
   - Calculado por build
   - Mesmo tratamento de edge cases (1 TC = APFD 1.0)

3. **Mesmos Critérios de Inclusão:**
   - Apenas builds com pelo menos 1 falha
   - Mesmo número de builds avaliados

4. **Testes Estatísticos:**
   - Wilcoxon signed-rank (pareado, não-paramétrico)
   - Effect size: Cliff's delta e Cohen's d
   - Intervalos de confiança bootstrap

## Referência do Método NodeRank

```
Li, Y., et al. (2024). Test Input Prioritization for Graph Neural Networks.
IEEE Transactions on Software Engineering, 50(5), 1178-1195.
DOI: 10.1109/TSE.2024.3385538
```

O NodeRank usa:
- Graph Structure Mutation (GSM)
- Node Feature Mutation (NFM)
- GNN Model Mutation (GMM)
- Ensemble Learning (LR, RF, XGBoost, LightGBM)

## Configuração

Edite `run_noderank_industry.py` para ajustar:

```python
CONFIG = {
    'n_gsm_mutations': 5,      # Graph Structure Mutations
    'n_nfm_mutations': 5,      # Node Feature Mutations
    'n_gmm_variants': 4,       # Model Variants
    'use_ensemble': True,      # Ensemble de classificadores
    'seed': 42,                # Seed para reproducibilidade
}
```

## Resultados Esperados

Após execução, você terá arquivos comparáveis:

| Arquivo | NodeRank | Filo-Priori |
|---------|----------|-------------|
| APFD por build | `results/noderank_industry/apfd_per_build_FULL_testcsv.csv` | `results/experiment_industry/apfd_per_build_FULL_testcsv.csv` |

Ambos seguem o formato:
```csv
method_name,build_id,test_scenario,count_tc,count_commits,apfd,time
```
