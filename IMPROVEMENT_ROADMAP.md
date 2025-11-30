# Roteiro de Melhorias: Filo-Priori V11

**Objetivo**: Superar DeepOrder (APFD 0.6500) mantendo as contribuições científicas do Filo-Priori

**Gap atual**: 0.6500 - 0.6413 = 0.0087 (1.34%)

---

## Fase 1: Análise das Vantagens do DeepOrder

### O que DeepOrder faz melhor:

1. **Online Learning**: Atualiza o modelo após cada build de teste
   - Filo-Priori: Treina offline em batch
   - DeepOrder: Adapta continuamente

2. **Histórico Ilimitado**: Usa toda a história do teste
   - Filo-Priori: Janela fixa de 5 builds
   - DeepOrder: Acumula todo histórico

3. **Features Essenciais**: Foca no que importa
   - `last_verdict`: Resultado da última execução
   - `failure_rate`: Taxa de falha histórica
   - `time_since_failure`: Recência da última falha
   - `consecutive_same`: Padrão de estabilidade

4. **Simplicidade Efetiva**: MLP simples evita overfitting
   - 3 camadas ocultas: [64, 32, 16]
   - Dropout 0.2
   - BatchNorm

---

## Fase 2: Melhorias Prioritárias (Alto Impacto)

### 2.1 Incorporar Online Learning ao Filo-Priori

**Implementação**:
```python
# Após cada build de validação/teste:
1. Fazer predição
2. Calcular APFD
3. Atualizar feature extractor com resultados reais
4. Fine-tune modelo (1-2 épocas) nos últimos N builds
```

**Impacto esperado**: +1-2% APFD

### 2.2 Expandir Histórico Estrutural

**Atual**: `recent_window = 5`
**Proposto**: Usar toda história com decay exponencial

```python
# Weighted features com decay temporal
weight = exp(-alpha * (current_build - execution_build))
failure_rate_weighted = sum(weight * verdict) / sum(weight)
```

**Impacto esperado**: +0.5-1% APFD

### 2.3 Adicionar Features do DeepOrder que Faltam

**Features a adicionar**:
1. `last_verdict` (0/1) - Já temos implícito, mas não explícito
2. `time_since_last_failure` (normalizado)
3. `execution_frequency` (execuções / builds)
4. `avg_position_in_build` (posição média na ordem original)

**Impacto esperado**: +0.5-1% APFD

### 2.4 Otimizar Pesos do Grafo (Learned Edge Weights)

**Atual**: Pesos fixos (1.0, 0.5, 0.3)
**Proposto**: Aprender pesos via attention ou parametrização

```python
# Edge attention weights
edge_weights = sigmoid(MLP([edge_type_embedding, node_features]))
```

**Impacto esperado**: +0.5-1% APFD

---

## Fase 3: Melhorias Secundárias (Médio Impacto)

### 3.1 Gated Fusion ao invés de Cross-Attention

**Atual**: Concatenação após cross-attention
**Proposto**: Gate adaptativo por amostra

```python
gate = sigmoid(Linear(concat([semantic, structural])))
fused = gate * semantic + (1 - gate) * structural
```

**Benefício**: Melhor handling de testes novos (sem histórico)

### 3.2 Aumentar Capacidade do Structural Stream

**Atual**: 10 features → GAT 2 camadas
**Proposto**: Expandir para 16-20 features selecionadas automaticamente

### 3.3 Ranking Loss Mais Agressivo

**Atual**: 0.7 * focal + 0.3 * ranking
**Proposto**: 0.5 * focal + 0.5 * ranking (após warmup)

---

## Fase 4: Configuração Experimental

### 4.1 Experimento A: Online Learning

```yaml
online_learning:
  enabled: true
  update_frequency: every_build
  fine_tune_epochs: 2
  learning_rate: 1e-5  # Lower for fine-tuning
```

### 4.2 Experimento B: Extended Features

```yaml
structural:
  features:
    - failure_rate        # histórico completo
    - recent_failure_rate # últimos 5 builds
    - last_verdict        # NOVO: última execução
    - time_since_failure  # NOVO: recência
    - execution_frequency # NOVO: frequência
    - consecutive_failures
    - failure_trend
    - flakiness_rate
    - test_novelty
    - weighted_failure_rate  # NOVO: com decay
```

### 4.3 Experimento C: Learned Graph Weights

```yaml
graph:
  edge_weight_learning: true
  initial_weights:
    co_failure: 1.0
    co_success: 0.5
    semantic: 0.3
  learnable: true
```

### 4.4 Experimento D: Gated Fusion

```yaml
fusion:
  type: gated  # ao invés de cross_attention
  gate_activation: sigmoid
```

---

## Fase 5: Ordem de Execução

### Sprint 1: Quick Wins (1-2 horas)
1. ✅ Adicionar features `last_verdict`, `time_since_failure`
2. ✅ Aumentar `recent_window` de 5 para 10
3. ✅ Testar ranking loss 0.5/0.5

### Sprint 2: Online Learning (2-3 horas)
1. ✅ Implementar atualização de features após cada build
2. ✅ Fine-tuning incremental
3. ✅ Validar no dataset

### Sprint 3: Graph Optimization (2-3 horas)
1. ✅ Implementar learned edge weights
2. ✅ Testar diferentes inicializações
3. ✅ Ablation study

### Sprint 4: Fusion Refinement (1-2 horas)
1. ✅ Ativar GatedFusion
2. ✅ Comparar com cross-attention
3. ✅ Selecionar melhor

---

## Fase 6: Critérios de Sucesso

### Métricas Alvo
- **APFD** ≥ 0.6550 (superar DeepOrder por margem significativa)
- **p-value** < 0.05 (diferença estatisticamente significativa)
- **Consistência**: Melhoria em ≥80% dos builds

### Validação
1. Executar 5 runs com seeds diferentes
2. Calcular média e desvio padrão
3. Wilcoxon signed-rank test vs DeepOrder

---

## Fase 7: Plano de Implementação

### Arquivo a Modificar
1. `src/preprocessing/structural_feature_extractor_v2_5.py` - Adicionar features
2. `src/models/dual_stream_v8.py` - Gated fusion opcional
3. `configs/experiment_industry_v11.yaml` - Nova configuração
4. `main.py` - Online learning loop

### Ordem de Commits
1. `feat(features): add last_verdict and time_since_failure features`
2. `feat(config): increase recent_window to 10`
3. `feat(loss): adjust ranking loss weight to 0.5`
4. `feat(online): implement online feature update`
5. `feat(fusion): add gated fusion option`
6. `experiment: run v11 comparison vs DeepOrder`

---

## Resumo Executivo

| Melhoria | Impacto Esperado | Esforço | Prioridade |
|----------|-----------------|---------|------------|
| Online Learning | +1-2% | Médio | **Alta** |
| Extended Features | +0.5-1% | Baixo | **Alta** |
| Histórico com Decay | +0.5-1% | Baixo | **Alta** |
| Learned Edge Weights | +0.5-1% | Médio | Média |
| Gated Fusion | +0.5-1% | Baixo | Média |
| Ranking Loss 0.5 | +0.3-0.5% | Mínimo | Alta |

**Total esperado**: +2.3% a +5.5% APFD

**Meta**: APFD ≥ 0.6550 (vs DeepOrder 0.6500)
