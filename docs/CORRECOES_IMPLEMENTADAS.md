# Correções Implementadas - Filo-Priori V8

**Data**: 2025-11-07
**Status**: 🟡 **PRONTO PARA TESTE**

---

## 🎯 RESUMO EXECUTIVO

Foram identificados e corrigidos **5 problemas críticos** que causaram o colapso de predição no modelo V8:

1. ✅ Focal Loss alpha INVERTIDO
2. ✅ Binary strategy INCORRETO (pass_vs_all → pass_vs_fail)
3. ✅ Learning rate muito baixo
4. ✅ Regularização excessiva (dropout, weight_decay)
5. ✅ Early stopping muito agressivo

**Arquivo corrigido**: `configs/experiment_v8_fixed.yaml`

---

## 🔴 PROBLEMA CRÍTICO #1: Binary Strategy Incorreto

### Descoberta

**Feedback do Usuário**: "Não, não é pra incluir Delete/Blocked apenas 'Fail'.. 'Pass' e 'Fail'"

O modelo deve classificar **APENAS Pass vs Fail**, excluindo todas as outras classes.

### Antes (❌ INCORRETO)

```yaml
data:
  binary_strategy: "pass_vs_all"
  binary_negative_class: "Not-Pass"  # Incluía Delete, Blocked, etc.
```

**Dataset**:
- Total: 69,169 amostras
- Pass: 61,224 (88.5%)
- Not-Pass: 7,945 (11.5%) ← Incluía Fail + Delete + Blocked + Conditional Pass + Pending
- Ratio: 7.7:1

### Agora (✅ CORRETO)

```yaml
data:
  binary_strategy: "pass_vs_fail"  # ✅ MUDANÇA CRÍTICA
  binary_negative_class: "Fail"  # APENAS Fail
```

**Dataset**:
- Total: ~62,878 amostras
- Pass: 61,224 (97.4%)
- Fail: ~1,654 (2.6%)
- Ratio: 37:1 ⚠️ **5x mais desbalanceado!**

**Amostras excluídas** (~6,291):
- Delete: 3,653
- Blocked: 1,862
- Conditional Pass: 654
- Pending: 116
- Outros: ~6

---

## 🔴 PROBLEMA CRÍTICO #2: Focal Loss Alpha Invertido

### Explicação do Focal Loss

```python
# Focal Loss formula:
loss = -alpha * (1 - p)^gamma * log(p)

# alpha[0] = peso para classe 0 (Fail)
# alpha[1] = peso para classe 1 (Pass)
```

**Classes minoritárias precisam de alpha ALTO**, não baixo!

### Antes (❌ ERRADO)

```yaml
loss:
  focal:
    alpha: [0.15, 0.85]  # Fail: 0.15 ❌, Pass: 0.85 ❌
    gamma: 2.0
```

**Problema**: Deu MAIS peso à classe majoritária (Pass) e MENOS à minoritária (Fail)!

### Agora (✅ CORRETO)

```yaml
loss:
  focal:
    alpha: [0.995, 0.005]  # Fail: 0.995 ✅, Pass: 0.005 ✅
    gamma: 3.5  # Aumentado de 2.0
```

**Resultado**:
- Fail (2.6% do dataset) recebe peso 0.995
- Pass (97.4% do dataset) recebe peso 0.005
- Ratio: 199:1 no alpha (vs 37:1 no dataset)

---

## 🔧 OUTRAS CORREÇÕES

### 3. Learning Rate

```yaml
# Antes
learning_rate: 5e-5  # Muito conservador

# Agora
learning_rate: 1e-4  # 2x maior
```

### 4. Regularização

```yaml
# Antes
weight_decay: 2e-4
dropout: 0.3-0.4

# Agora
weight_decay: 5e-5  # 4x menor
dropout: 0.2-0.3  # Reduzido
```

### 5. Early Stopping

```yaml
# Antes
early_stopping:
  patience: 12

# Agora
early_stopping:
  patience: 20
  min_delta: 0.001  # Adicionar
```

### 6. Threshold Search

```yaml
# Antes
threshold_search:
  search_range: [0.2, 0.8]

# Agora
threshold_search:
  search_range: [0.1, 0.9]  # Mais amplo
```

---

## 📊 COMPARAÇÃO: ANTES vs DEPOIS

| Aspecto | Antes (❌) | Agora (✅) | Mudança |
|---------|-----------|-----------|---------|
| **Binary Strategy** | pass_vs_all | pass_vs_fail | ✅ CRÍTICO |
| **Dataset Size** | 69,169 | 62,878 | -9.1% |
| **Class Ratio** | 88.5%/11.5% (7.7:1) | 97.4%/2.6% (37:1) | 5x mais desbalanceado |
| **Focal Alpha (Fail)** | 0.15 | 0.995 | 6.6x maior |
| **Focal Alpha (Pass)** | 0.85 | 0.005 | 170x menor |
| **Focal Gamma** | 2.0 | 3.5 | +75% |
| **Learning Rate** | 5e-5 | 1e-4 | 2x maior |
| **Weight Decay** | 2e-4 | 5e-5 | 4x menor |
| **Dropout** | 0.3-0.4 | 0.2-0.3 | -25% |
| **Early Stop Patience** | 12 | 20 | +67% |

---

## 🎯 MÉTRICAS ESPERADAS

### Antes (Resultado Real)

```
Test Accuracy: 90.39%
Test F1 Macro: 0.4748
Test APFD: 0.4969

Classification Report:
              precision    recall  f1-score   support
    Not-Pass       0.00      0.00      0.00       781  ❌
    Pass           0.90      1.00      0.95      7346  ✓
```

**Problema**: Colapso de predição - modelo NUNCA detecta falhas!

### Agora (Esperado)

```
Test Accuracy: ≥80%
Test F1 Macro: ≥0.50
Test APFD: ≥0.55

Classification Report:
              precision    recall  f1-score   support
    Fail           0.25      0.30      0.27       165  ✅
    Pass           0.98      0.97      0.98      6181  ✅
```

**Metas**:
- ✅ Recall Fail ≥ 30% (detectar pelo menos 30% das falhas)
- ✅ Precision Fail ≥ 25% (evitar muitos falsos alarmes)
- ✅ Prediction Diversity ≥ 0.20 (ambas classes sendo preditas)
- ✅ F1 Macro ≥ 0.50 (performance balanceada)

---

## 🚦 CRITÉRIOS DE SUCESSO

### ✅ GO (Sucesso)
- Prediction Diversity ≥ 0.20
- Recall Fail ≥ 0.30
- Precision Fail ≥ 0.25
- F1 Macro ≥ 0.50
- Test Accuracy ≥ 0.80

### ⚠️ REVIEW (Ajustes Necessários)
- 0.15 ≤ Prediction Diversity < 0.20
- 0.20 ≤ Recall Fail < 0.30
- 0.45 ≤ F1 Macro < 0.50

### ❌ NO-GO (Falha)
- Prediction Diversity < 0.15 (ainda colapso)
- Recall Fail < 0.20 (não detecta falhas)
- F1 Macro < 0.45 (pior que baseline)

---

## 📝 PRÓXIMOS PASSOS

### 1. Teste Rápido (RECOMENDADO)

```bash
# Teste com 5K amostras e 10 épocas (~15-20 minutos)
python main_v8.py --config configs/experiment_v8_fixed.yaml \
                   --sample-size 5000 \
                   --num-epochs 10
```

**Objetivo**: Validar que o modelo está aprendendo antes do treino completo.

**Verificar**:
- Prediction Diversity está aumentando?
- Recall Fail > 0.20?
- F1 Macro melhorando ao longo das épocas?

### 2. Treino Completo (SE TESTE RÁPIDO OK)

```bash
# Treino completo com dataset inteiro
python main_v8.py --config configs/experiment_v8_fixed.yaml
```

**Duração estimada**: 2-3 horas

### 3. Alternativa: Weighted CE (SE FOCAL FALHAR)

Se o teste rápido ainda mostrar colapso, considerar trocar Focal Loss por Weighted CE:

```yaml
loss:
  type: "weighted_ce"
  weighted_ce:
    use_class_weights: true
    class_weights: [37.0, 1.0]  # Ratio direto
```

---

## 🔍 MONITORAMENTO

Durante o treino, monitorar:

### 1. Diversidade de Predição (a cada época)
```python
unique_preds = len(np.unique(predictions))
if unique_preds < 2:
    print("⚠️ WARNING: Prediction collapse detected!")
```

### 2. Recall de Ambas as Classes
```
Epoch 5:
  - Recall Fail: 0.18 → 0.23 → 0.28 ✅ (melhorando)
  - Recall Pass: 0.98 → 0.97 → 0.96 ✅ (estável)
```

### 3. F1 Macro ao Longo das Épocas
```
Epoch 1: 0.47 (baseline)
Epoch 5: 0.51 ✅ (melhorando)
Epoch 10: 0.54 ✅ (convergindo)
```

### 4. Loss Diminuindo
```
Train Loss: 0.150 → 0.095 → 0.068 ✅
Val Loss: 0.162 → 0.108 → 0.082 ✅
```

---

## 📁 ARQUIVOS MODIFICADOS

1. **configs/experiment_v8_fixed.yaml** (criado)
   - Todas as correções implementadas
   - Pronto para uso

2. **ANALISE_PROBLEMAS_TREINAMENTO.md** (atualizado)
   - Problema #5: Binary Strategy documentado
   - Plano de ação atualizado

3. **CORRECOES_IMPLEMENTADAS.md** (este arquivo)
   - Resumo completo das mudanças
   - Guia de próximos passos

---

## 🎓 LIÇÕES APRENDIDAS

1. **Focal Loss Alpha é contra-intuitivo**:
   - Alpha NÃO é "peso da classe"
   - É "fator de down-weight para exemplos fáceis"
   - Classe minoritária precisa alpha ALTO (0.9-0.999)

2. **Binary Strategy importa**:
   - Pass vs Fail ≠ Pass vs Not-Pass
   - Semântica diferente, dataset diferente
   - Sempre validar com o usuário

3. **Imbalance extremo requer medidas extremas**:
   - 97%/3% é MUITO desbalanceado
   - Focal Loss precisa ser muito agressivo
   - alpha=[0.995, 0.005] não é exagero

4. **Threshold 0.5 não funciona para imbalanced**:
   - Sempre usar threshold search
   - Threshold ótimo provavelmente será ~0.1-0.3

5. **Monitoramento é crucial**:
   - Prediction diversity detecta colapso
   - F1 Macro estagnado = modelo não aprendeu
   - Verificar AMBAS as classes, não só accuracy

---

**Status**: 🟡 **CORREÇÕES IMPLEMENTADAS - PRONTO PARA TESTE**

**Próxima ação**: Executar teste rápido para validar correções antes do treino completo.

```bash
python main_v8.py --config configs/experiment_v8_fixed.yaml --sample-size 5000 --num-epochs 10
```
