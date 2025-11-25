# 📊 Plano de Expansão: Structural Features 6 → 29

**Objetivo**: Melhorar APFD de 0.6210 → 0.65-0.70 através de features mais ricas

**Data**: 2025-11-14
**Status**: Em implementação

---

## 🎯 Features Atuais (6)

| # | Feature | Tipo | Descrição |
|---|---------|------|-----------|
| 1 | test_age | Phylogenetic | Builds desde primeira aparição |
| 2 | failure_rate | Phylogenetic | Taxa histórica de falhas |
| 3 | recent_failure_rate | Phylogenetic | Taxa de falhas nos últimos N builds |
| 4 | flakiness_rate | Phylogenetic | Taxa de transições Pass↔Fail |
| 5 | commit_count | Structural | Número de commits/CRs |
| 6 | test_novelty | Structural | Flag de primeira aparição |

---

## 🚀 Features Novas (23 adicionais → Total 29)

### A. TEMPORAL/HISTORY FEATURES (10 novas)

| # | Feature | Fórmula/Lógica | Valor Esperado |
|---|---------|----------------|----------------|
| 7 | execution_count | Total de execuções | 1-1000+ |
| 8 | failure_count | Total de falhas | 0-50 |
| 9 | pass_count | Total de passes | 1-1000+ |
| 10 | consecutive_failures | Streak atual de falhas | 0-10 |
| 11 | consecutive_passes | Streak atual de passes | 0-100 |
| 12 | max_consecutive_failures | Maior streak de falhas | 0-20 |
| 13 | last_failure_age | Builds desde última falha | 0-500 |
| 14 | last_pass_age | Builds desde último pass | 0-500 |
| 15 | execution_frequency | executions / builds_span | 0.1-1.0 |
| 16 | builds_since_change | Builds desde último commit | 0-100 |

**Impacto Esperado**:
- Testes com consecutive_failures alto → **mais prioritários**
- Testes com last_failure_age baixo → **mais prioritários**
- execution_frequency alto → testes importantes

### B. RECENCY & TREND FEATURES (6 novas)

| # | Feature | Fórmula/Lógica | Valor Esperado |
|---|---------|----------------|----------------|
| 17 | failure_trend | (recent_rate - overall_rate) | -1.0 a +1.0 |
| 18 | recent_execution_count | Execuções nos últimos 5 builds | 0-5 |
| 19 | very_recent_failure_rate | Taxa nos últimos 2 builds | 0.0-1.0 |
| 20 | medium_term_failure_rate | Taxa nos últimos 10 builds | 0.0-1.0 |
| 21 | acceleration | (very_recent - recent) | -1.0 a +1.0 |
| 22 | deceleration_factor | recent / overall (if overall > 0) | 0.0-5.0 |

**Impacto Esperado**:
- failure_trend **positivo** → **falhas aumentando** → mais prioritário
- acceleration **positivo** → **aceleração de falhas** → muito prioritário

### C. BUILD/CHANGE FEATURES (4 novas)

| # | Feature | Fórmula/Lógica | Valor Esperado |
|---|---------|----------------|----------------|
| 23 | builds_affected | Unique builds com este teste | 1-500 |
| 24 | cr_count | Número de CRs (separado) | 0-10 |
| 25 | avg_commits_per_execution | commit_count / execution_count | 0.1-5.0 |
| 26 | recent_commit_surge | commits_recent > avg_commits * 1.5 | 0.0-1.0 (bool) |

**Impacto Esperado**:
- cr_count **alto** → mudanças recentes → mais prioritário
- recent_commit_surge **alto** → atividade recente → mais prioritário

### D. STABILITY/VOLATILITY FEATURES (3 novas)

| # | Feature | Fórmula/Lógica | Valor Esperado |
|---|---------|----------------|----------------|
| 27 | stability_score | 1.0 - flakiness_rate | 0.0-1.0 |
| 28 | pass_fail_ratio | pass_count / (failure_count + 1) | 0.1-100.0 |
| 29 | recent_stability | 1.0 - recent_flakiness | 0.0-1.0 |

**Impacto Esperado**:
- stability_score **baixo** → teste instável → mais prioritário
- pass_fail_ratio **baixo** → falha frequente → mais prioritário

---

## 📊 Feature Groups Summary

| Grupo | Features | Total | Objetivo |
|-------|----------|-------|----------|
| **ATUAL** | test_age, failure_rate, etc. | 6 | Baseline |
| **TEMPORAL** | execution_count, streaks, ages | 10 | Padrões históricos |
| **RECENCY** | trends, acceleration | 6 | Detecção de mudanças recentes |
| **CHANGE** | builds, CRs, commits | 4 | Impacto de mudanças de código |
| **STABILITY** | volatility, ratios | 3 | Confiabilidade |
| **TOTAL** | | **29** | Caracterização completa |

---

## 🔧 Implementação

### 1. Modificar `StructuralFeatureExtractor`

**Arquivo**: `src/preprocessing/structural_feature_extractor.py`

**Mudanças**:
1. Expandir `_extract_phylogenetic_features()` → retornar 20 features
2. Expandir `_extract_structural_features()` → retornar 9 features
3. Atualizar `get_feature_names()` → listar 29 features
4. Adicionar métodos auxiliares:
   - `_compute_streaks()` - consecutive failures/passes
   - `_compute_trends()` - failure trends
   - `_compute_change_features()` - CR/commit analysis

### 2. Atualizar Configurações

**Arquivo**: `configs/experiment_05_expanded_features.yaml`

```yaml
structural:
  input_dim: 29  # ← de 6 para 29
  extractor:
    recent_window: 5
    very_recent_window: 2
    medium_term_window: 10
    min_history: 2
    cache_path: "cache/structural_features_v2.pkl"
```

### 3. Atualizar Modelo

**Arquivo**: `src/models/dual_stream_v8.py`

**Mudanças**: Nenhuma! O modelo já aceita `input_dim` variável.

Apenas ajustar config:
```yaml
model:
  structural:
    input_dim: 29  # ← CRITICAL: update from 6!
    hidden_dim: 128  # ↑ aumentar de 64 (mais features precisam mais capacity)
```

---

## 📈 Impacto Esperado

### Antes (6 features)

```
APFD: 0.6210
F1 Macro: 0.5294
Recall Pass: 0.99
Recall Fail: 0.05
```

### Depois (29 features) - ESPERADO

```
APFD: 0.65-0.70  (+5-13% melhoria)
F1 Macro: 0.55-0.60  (+4-13% melhoria)
Recall Pass: 0.99  (mantém)
Recall Fail: 0.08-0.15  (+60-200% melhoria)
```

### Por Que Vai Melhorar?

1. **Trends detectam mudanças recentes**
   - Tests com `failure_trend > 0` → começando a falhar
   - Tests com `acceleration > 0` → falhas acelerando

2. **Streaks capturam padrões**
   - `consecutive_failures = 3` → provável falhar de novo
   - `max_consecutive_failures = 10` → teste problemático

3. **Change features ligam código → falhas**
   - `recent_commit_surge = 1` → mudanças recentes → risco
   - `cr_count alto` → muitas mudanças → risco

4. **Stability features identificam flaky tests**
   - `stability_score < 0.5` → teste instável → prioritário

---

## 🎯 Experimento 05: Expanded Features

### Config

```yaml
experiment:
  name: "experiment_05_expanded_features"
  version: "5.0"
  description: "29 structural features (de 6) - melhoria de ranking"

structural:
  input_dim: 29
  extractor:
    recent_window: 5
    very_recent_window: 2
    medium_term_window: 10

model:
  structural:
    input_dim: 29
    hidden_dim: 128  # ↑ de 64 (dobro de features → dobro de hidden)
    num_layers: 2
    dropout: 0.1

# Rest identical to 04a (Weighted CE vencedor)
```

### Critérios de Sucesso

| Métrica | Exp 04a (Baseline) | Exp 05 (Target) | Melhoria |
|---------|-------------------|----------------|----------|
| **APFD** | 0.6210 | **≥ 0.65** | **+5%** |
| F1 Macro | 0.5294 | ≥ 0.55 | +4% |
| Recall Pass | 0.99 | ≥ 0.98 | mantém |
| Recall Fail | 0.05 | ≥ 0.08 | +60% |

**Critério GO/NO-GO**:
- ✅ **GO**: APFD ≥ 0.63 (melhoria de +2%)
- ❌ **NO-GO**: APFD < 0.62 (pior ou igual)

---

## 📝 Checklist de Implementação

### Fase 1: Código (2-3 horas)

- [ ] Criar `structural_feature_extractor_v2.py` com 29 features
- [ ] Adicionar métodos auxiliares (_compute_streaks, _compute_trends)
- [ ] Atualizar `get_feature_names()` com 29 nomes
- [ ] Testar extração com sample de dados

### Fase 2: Configuração (30 min)

- [ ] Criar `configs/experiment_05_expanded_features.yaml`
- [ ] Atualizar `structural.input_dim: 29`
- [ ] Atualizar `model.structural.input_dim: 29`
- [ ] Aumentar `model.structural.hidden_dim: 128`

### Fase 3: Validação (30 min)

- [ ] Executar com `--sample-size 500` para validação rápida
- [ ] Verificar shapes: (batch, 29) features
- [ ] Confirmar sem erros de dimensão
- [ ] Conferir feature statistics (mean, std)

### Fase 4: Experimento Completo (2-3 horas)

- [ ] Executar `experiment_05` no dataset completo
- [ ] Monitorar training (50 épocas)
- [ ] Comparar APFD com 04a
- [ ] Decisão GO/NO-GO

---

## 🚨 Riscos e Mitigações

### Risco 1: Overfitting com 29 Features

**Mitigação**:
- Aumentar dropout: 0.1 → 0.15
- Aumentar weight_decay: 1e-4 → 2e-4
- Monitorar val-test gap

### Risco 2: Features Redundantes

**Mitigação**:
- Correlação entre features será absorvida pelo modelo
- Dropout ajuda com redundância
- Não é problema crítico para GATs

### Risco 3: Aumento de Tempo de Computação

**Impacto**: Mínimo!
- Feature extraction: +10% tempo (mais features)
- Model forward: +5% tempo (29 vs 6 input)
- Total: ~+15% tempo (aceitável)

**Mitigação**: Usar caching de features

---

## 📊 Análise de Importância de Features (Post-Hoc)

Após experimento 05, podemos analisar feature importance:

1. **Gradient-based importance** - quais features têm maior gradiente
2. **Ablation study** - remover features e ver impacto
3. **SHAP values** - contribuição de cada feature

**Objetivo**: Identificar top-10 features mais importantes para simplificar modelo futuro

---

**Próxima Ação**: Implementar `StructuralFeatureExtractorV2` com 29 features

**Tempo Estimado Total**: 6-8 horas (código + teste + experimento)

---

**Autor**: Claude Code
**Data**: 2025-11-14
**Status**: 📝 PLANO COMPLETO - Pronto para implementação
