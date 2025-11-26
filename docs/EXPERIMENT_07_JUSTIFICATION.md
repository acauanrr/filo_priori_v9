# Experimento 07: Ranking-Optimized Filo-Priori

## Problema Identificado

O modelo atual (Exp 06) atinge APFD = 0.6171, **perdendo** para o baseline simples FailureRate (APFD = 0.6289). Isso é inaceitável para um modelo de Deep Learning com:
- GATv2 attention
- Dual-stream architecture
- Multi-edge phylogenetic graph
- 10 features estruturais selecionadas

## Análise da Causa Raiz

### Desalinhamento Fundamental: Classificação vs Ranking

| Aspecto | Modelo Atual | APFD Requer |
|---------|--------------|-------------|
| **Otimização** | Cross-entropy (classificação) | Ranking loss |
| **Objetivo** | Minimizar erro de predição | Ordenar failures antes de passes |
| **Métrica** | Accuracy, F1 | Posição relativa das falhas |

**Insight Crítico**: Um modelo com 99% accuracy pode ter APFD péssimo se ranquear todas as falhas por último!

### Por que FailureRate Ganha?

FailureRate usa uma heurística simples: `P(Fail) = num_failures / num_executions`

Isso funciona porque:
1. **Alinhamento direto com APFD**: Testes que falharam frequentemente são ranqueados primeiro
2. **Sem problemas de calibração**: A probabilidade É a taxa de falha
3. **Robust a outliers**: Laplace smoothing estabiliza estimativas

## Solução Proposta: Ranking-Aware Training

### 1. Ranking Loss (RankNet-Style)

**Referência**: Burges et al. "Learning to Rank using Gradient Descent" (ICML 2005)

```
L_ranking = Σ softplus(-(s_fail - s_pass))
```

Onde:
- `s_fail` = score do test case que falhou
- `s_pass` = score do test case que passou

**Intuição**: Penaliza quando um Pass tem score maior que um Fail do mesmo build.

### 2. Hard Negative Mining

**Referência**: He et al. "Deep Residual Learning" (CVPR 2016)

Em vez de comparar com todos os Passes, focamos nos mais difíceis:
- Top-K passes com maior P(Fail) predito
- Estes são os "hard negatives" que confundem o modelo

### 3. Grouped Batch Sampling

**Intuição**: Para calcular ranking loss, precisamos de pares (Fail, Pass) do mesmo build.

O sampler agrupa amostras por Build_ID, garantindo que cada batch contenha testes do mesmo build.

### 4. Weighted Focal Loss

**Referência**: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)

```
L_focal = -α(1-p_t)^γ * log(p_t)
```

- `α = 0.75`: Peso para classe minoritária (Fail)
- `γ = 2.5`: Foco em hard examples

## Configuração do Experimento 07

### Mudanças Críticas

| Parâmetro | Exp 06 (Atual) | Exp 07 (Novo) |
|-----------|----------------|---------------|
| `ranking.enabled` | false | **true** |
| `ranking.use_grouped_sampler` | N/A | **true** |
| `ranking.weight` | N/A | **0.3** |
| `loss.type` | weighted_ce | **weighted_focal** |
| `loss.focal_alpha` | N/A | **0.75** |
| `loss.focal_gamma` | N/A | **2.5** |
| `threshold_search.enabled` | false | **true** |

### Impacto Esperado

| Métrica | Exp 06 | Exp 07 Target | Melhoria |
|---------|--------|---------------|----------|
| APFD | 0.6171 | **0.65-0.68** | +5-10% |
| vs FailureRate | Perde | **Ganha** | - |
| vs Random | +10.3% | **+16-22%** | +6-12pp |

## Como Executar

```bash
python main.py --config configs/experiment_07_ranking_optimized.yaml
```

Resultados em: `results/experiment_07_ranking_optimized/`

## Métricas de Sucesso

O experimento será considerado **bem-sucedido** se:

1. **APFD > 0.6289** (supera FailureRate)
2. **APFD > 0.63** (margem de segurança)
3. **p-value < 0.05** (estatisticamente significativo vs FailureRate)

## Referências Científicas

1. Burges, C. et al. (2005). "Learning to Rank using Gradient Descent." ICML.
2. Lin, T.Y. et al. (2017). "Focal Loss for Dense Object Detection." ICCV.
3. He, K. et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
4. Brody, S. et al. (2022). "How Attentive are Graph Attention Networks?" ICLR.

---

**Data**: 2025-11-26
**Autor**: Acauan C. Ribeiro
**Status**: Pronto para execução
