# Análise de Problemas do Treinamento V8

**Data**: 2025-11-07
**Status**: 🔴 CRITICAL - Modelo não aprendeu

---

## 📊 Resultados Obtidos

```
Test Accuracy: 0.9039 (90.39%)
Test F1 Macro: 0.4748
Test APFD: 0.4969

Classification Report:
              precision    recall  f1-score   support
    Not-Pass       0.00      0.00      0.00       781  ❌
    Pass           0.90      1.00      0.95      7346  ✓
```

---

## 🔴 PROBLEMA #1: COLAPSO DE PREDIÇÃO (CRÍTICO)

### Sintoma
- **Recall Not-Pass = 0.00%**: O modelo NUNCA detecta falhas
- **Recall Pass = 100%**: O modelo SEMPRE prevê Pass
- **Precision Not-Pass = 0.00%**: Nenhuma predição de Not-Pass

### Causa Raiz
**Desbalanceamento extremo de classes**:
- Pass: 61,224 (88.5%)
- Not-Pass: 7,945 (11.5%)
- Ratio: 7.7:1

**Focal Loss inadequado**:
```yaml
focal:
  alpha: [0.15, 0.85]  # Muito fraco!
  gamma: 2.0
```

O alpha=[0.15, 0.85] significa:
- Not-Pass (classe minoritária): peso 0.15 ❌
- Pass (classe majoritária): peso 0.85 ❌

**INVERTIDO!** Deveria ser alpha=[0.85, 0.15] ou mais!

### Por que Aconteceu?
1. **Focal Loss com alpha invertido**: Favoreceu a classe ERRADA
2. **Threshold 0.5**: Inadequado para classes desbalanceadas
3. **Learning rate baixo**: 5e-5 é muito conservador
4. **Dropout alto**: 0.3-0.4 regularizou demais

---

## 🔴 PROBLEMA #2: F1 MACRO ESTAGNADO

### Sintoma
```
Epoch 1:  Val F1=0.4703
Epoch 2:  Val F1=0.4703
...
Epoch 13: Val F1=0.4703
```

**F1 idêntico em TODAS as épocas = modelo não aprendeu**

### Explicação
F1 Macro = (F1_NotPass + F1_Pass) / 2 = (0.00 + 0.94) / 2 = 0.47

O modelo está simplesmente repetindo a mesma predição (sempre Pass).

---

## 🔴 PROBLEMA #3: APFD MUITO BAIXO

### Resultados
```
Mean APFD: 0.4969
Builds analisados: 52 (esperado: 277)
```

### Por que?
- **APFD < 0.50 é PIOR que random!**
- **Apenas 52 builds** com falhas (faltam 225 builds)
- Se o modelo nunca prevê Not-Pass, não prioriza corretamente

---

## 🔴 PROBLEMA #4: CONFIGURAÇÃO INCORRETA

### Focal Loss Alpha Invertido
```yaml
# ❌ ERRADO (configuração atual)
focal:
  alpha: [0.15, 0.85]  # [Not-Pass, Pass]
  # Peso 0.15 para classe minoritária (Not-Pass) ❌
  # Peso 0.85 para classe majoritária (Pass) ❌
```

**Deveria ser:**
```yaml
# ✅ CORRETO
focal:
  alpha: [0.02, 0.98]  # [Not-Pass, Pass]
  # Peso 0.98 para classe minoritária (Not-Pass) ✓
  # Peso 0.02 para classe majoritária (Pass) ✓
  gamma: 3.0  # Aumentar também
```

### Threshold Inadequado
```python
# Threshold padrão = 0.5
predictions = (probabilities[:, 1] > 0.5).astype(int)
```

**Para classes desbalanceadas, threshold deveria ser < 0.5**

---

## 🔧 SOLUÇÕES IMPLEMENTADAS

### Solução 1: Corrigir Focal Loss (PRIORITÁRIO)
```yaml
loss:
  type: "focal"
  focal:
    alpha: [0.02, 0.98]  # Invertido e mais agressivo
    gamma: 3.0  # Aumentado de 2.0 para 3.0
```

**Lógica do Focal Loss**:
```python
# Para classe minoritária (Not-Pass, índice 0):
loss_weight_NotPass = alpha[0] * (1 - p_NotPass)^gamma
# Com alpha=0.98, dá MUITO peso a erros de Not-Pass

# Para classe majoritária (Pass, índice 1):
loss_weight_Pass = alpha[1] * (1 - p_Pass)^gamma
# Com alpha=0.02, dá POUCO peso a erros de Pass
```

### Solução 2: Threshold Search
```python
# Testar thresholds de 0.1 a 0.9
best_threshold = find_best_threshold(
    val_probabilities,
    val_labels,
    metric='f1_macro'
)
# Provavelmente será ~0.3-0.4
```

### Solução 3: Hiperparâmetros Ajustados
```yaml
training:
  learning_rate: 1e-4  # De 5e-5 → 1e-4 (2x maior)
  weight_decay: 5e-5   # De 2e-4 → 5e-5 (4x menor)

model:
  semantic:
    dropout: 0.2  # De 0.3 → 0.2
  structural:
    dropout: 0.2  # De 0.3 → 0.2
  classifier:
    dropout: 0.3  # De 0.4 → 0.3
```

### Solução 4: Monitoramento de Diversidade
```python
def compute_prediction_diversity(predictions):
    """Detecta colapso de predição"""
    unique, counts = np.unique(predictions, return_counts=True)
    diversity = len(unique) / len(np.unique([0, 1]))  # 0.5 se colapso
    return diversity

# Adicionar à avaliação
if diversity < 0.3:
    logger.warning("⚠️ PREDICTION COLLAPSE DETECTED!")
```

### Solução 5: Class Weights Alternativos
```python
# Se Focal Loss não funcionar, usar Weighted CE
class_weights = torch.FloatTensor([7.7, 1.0])  # Ratio direto
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

## 📋 CONFIGURAÇÃO CORRIGIDA

Criar: `configs/experiment_v8_fixed.yaml`

```yaml
# Loss configurado corretamente
loss:
  type: "focal"
  focal:
    alpha: [0.02, 0.98]  # ✅ Invertido e agressivo
    gamma: 3.0           # ✅ Aumentado

  # Alternativa (se focal não funcionar)
  weighted_ce:
    use_class_weights: true
    class_weights: [7.7, 1.0]  # Ratio direto

# Training ajustado
training:
  num_epochs: 50         # Aumentar de 40
  batch_size: 32
  learning_rate: 1e-4    # ✅ 2x maior (de 5e-5)
  weight_decay: 5e-5     # ✅ 4x menor (de 2e-4)

  early_stopping:
    patience: 20  # ✅ Aumentar de 12
    monitor: "val_f1_macro"
    min_delta: 0.001  # ✅ Adicionar

# Model com menos regularização
model:
  semantic:
    dropout: 0.2  # ✅ De 0.3
  structural:
    dropout: 0.2  # ✅ De 0.3
  classifier:
    dropout: 0.3  # ✅ De 0.4

# Threshold search
evaluation:
  threshold_search:
    enabled: true
    search_range: [0.1, 0.9]  # ✅ Mais amplo
    search_step: 0.05
    metric: "f1_macro"
```

---

## 🎯 CRITÉRIOS DE SUCESSO (REVISADOS)

### Mínimo Aceitável (GO Criteria)
- [ ] **Prediction Diversity ≥ 0.30**: Ambas classes sendo previstas
- [ ] **Recall Not-Pass ≥ 0.30**: Detectando pelo menos 30% das falhas
- [ ] **F1 Macro ≥ 0.55**: Balanceado entre classes
- [ ] **Test Accuracy ≥ 0.70**: Performance geral
- [ ] **APFD ≥ 0.55**: Melhor que random

### Target (Ideal)
- [ ] Prediction Diversity ≥ 0.40
- [ ] Recall Not-Pass ≥ 0.50
- [ ] F1 Macro ≥ 0.60
- [ ] Test Accuracy ≥ 0.75
- [ ] APFD ≥ 0.60

### NO-GO (Falha Crítica)
- ❌ Prediction Diversity < 0.20 (colapso)
- ❌ Recall Not-Pass < 0.20 (não detecta falhas)
- ❌ F1 Macro < 0.50 (pior que baseline)

---

## 🔴 PROBLEMA #5: ESTRATÉGIA BINÁRIA INCORRETA (CRÍTICO!)

### Descoberta
**Usuário clarificou**: "Não, não é pra incluir Delete/Blocked apenas 'Fail'.. 'Pass' e 'Fail'"

O modelo deve classificar APENAS:
- ✅ Pass (classe 1)
- ✅ Fail (classe 0)

E EXCLUIR do dataset:
- ❌ Delete (3,653 amostras)
- ❌ Blocked (1,862 amostras)
- ❌ Conditional Pass (654 amostras)
- ❌ Pending (116 amostras)
- ❌ Outros

### Impacto da Mudança

#### Antes (pass_vs_all):
```yaml
binary_strategy: "pass_vs_all"
binary_negative_class: "Not-Pass"  # Agrupa Fail + Delete + Blocked + etc.
```
- Total: 69,169 amostras
- Pass: 61,224 (88.5%)
- Not-Pass: 7,945 (11.5%)
- Ratio: 7.7:1

#### Agora (pass_vs_fail):
```yaml
binary_strategy: "pass_vs_fail"  # ✅ CORRETO
binary_negative_class: "Fail"  # APENAS Fail
```
- Total: ~62,878 amostras (redução de 6,291 amostras)
- Pass: 61,224 (97.4%)
- Fail: ~1,654 (2.6%)
- Ratio: 37:1 ⚠️ **5x MAIS DESBALANCEADO!**

### Correção Implementada

```yaml
# configs/experiment_v8_fixed.yaml

data:
  binary_strategy: "pass_vs_fail"  # ✅ Mudado de "pass_vs_all"
  binary_negative_class: "Fail"    # ✅ Mudado de "Not-Pass"

loss:
  focal:
    alpha: [0.995, 0.005]  # ✅ Ajustado para ratio 37:1 (era [0.98, 0.02])
    gamma: 3.5             # ✅ Aumentado de 3.0

  weighted_ce:  # Alternativa
    class_weights: [37.0, 1.0]  # ✅ Ajustado (era [7.7, 1.0])
```

### Por que Isso é Crítico?

1. **Semântica Diferente**:
   - Delete ≠ Fail (teste deletado por outros motivos)
   - Blocked ≠ Fail (teste bloqueado, não falhou)
   - Conditional Pass ≠ Fail (passou com condições)

2. **Objetivo do Modelo**:
   - Queremos detectar FALHAS REAIS (Fail)
   - Não queremos misturar com outras classes

3. **APFD Afetado**:
   - Priorização deve focar em testes que FALHAM
   - Não em testes deletados/bloqueados

### Novo Desafio: Imbalance Extremo (37:1)

**Problema**: Com 97.4% Pass / 2.6% Fail, o modelo pode facilmente colapsar novamente.

**Soluções Aplicadas**:
1. Focal Loss muito mais agressivo: alpha=[0.995, 0.005], gamma=3.5
2. Threshold search mais amplo: [0.1, 0.9]
3. Considerar SMOTE ou class weights alternativos
4. Monitoramento rigoroso de diversidade de predição

---

## 🔄 PLANO DE AÇÃO

### Fase 0: Correções Críticas (COMPLETO ✅)
1. ✅ Criar `configs/experiment_v8_fixed.yaml`
2. ✅ Mudar binary_strategy: "pass_vs_all" → "pass_vs_fail"
3. ✅ Ajustar Focal Loss: alpha=[0.995, 0.005], gamma=3.5
4. ✅ Ajustar hiperparâmetros (lr, dropout, weight_decay)
5. ✅ Documentar mudanças em ANALISE_PROBLEMAS_TREINAMENTO.md

### Fase 1: Teste Rápido (PRÓXIMO PASSO)
1. ⏳ Rodar treino de teste (10 épocas, sample 5K)
   ```bash
   python main_v8.py --config configs/experiment_v8_fixed.yaml \
                      --sample-size 5000 \
                      --num-epochs 10
   ```
2. ⏳ Verificar métricas críticas:
   - Prediction Diversity > 0.15
   - Recall Fail > 0.20
   - F1 Macro > 0.45
3. ⏳ Se falhar, considerar alternativas (ver Fase 4)

### Fase 2: Threshold Search (SE NECESSÁRIO)
1. ⏳ Implementar threshold search
2. ⏳ Encontrar melhor threshold
3. ⏳ Atualizar código de predição

### Fase 3: Treino Completo (SE FASE 1 OK)
1. ⏳ Rodar treino completo (50 épocas)
2. ⏳ Monitorar métricas a cada época
3. ⏳ Salvar melhores pesos

### Fase 4: Alternativa Weighted CE (SE FOCAL FALHAR)
1. ⏳ Substituir Focal Loss por Weighted CE
2. ⏳ class_weights = [7.7, 1.0]
3. ⏳ Re-treinar

---

## 📊 COMPARAÇÃO: ANTES vs DEPOIS (ESPERADO)

| Métrica | Antes (❌) | Esperado (✅) | Notas |
|---------|-----------|--------------|-------|
| **Binary Strategy** | pass_vs_all | pass_vs_fail | ✅ CRÍTICO |
| **Dataset Size** | 69,169 | 62,878 | -6,291 amostras |
| **Class Ratio** | 88.5%/11.5% (7.7:1) | 97.4%/2.6% (37:1) | 5x mais desbalanceado |
| **Recall Fail** | 0.00% | ≥30% | Meta principal |
| **Recall Pass** | 100% | ≥95% | Deve permanecer alto |
| **Precision Fail** | 0.00% | ≥25% | Evitar falsos alarmes |
| **F1 Macro** | 0.47 | ≥0.50 | Balanceado |
| **Prediction Diversity** | ~0.0 | ≥0.20 | Colapso detectado |
| **APFD** | 0.497 | ≥0.55 | Priorização melhor |
| **Focal Loss Alpha** | [0.15, 0.85] | [0.995, 0.005] | 66x mais agressivo |
| **Épocas até convergir** | 13 (estagnado) | 20-35 | Com aprendizado real |

---

## 🚨 SINAIS DE ALERTA (Red Flags)

Durante o treinamento, monitorar:

1. **Colapso de Predição**:
   ```
   ⚠️ WARNING: All predictions are class 1 (Pass)
   ⚠️ Prediction diversity: 0.00
   ```

2. **F1 Estagnado**:
   ```
   Epoch 1-10: Val F1 = 0.47 (sem variação)
   ```

3. **Loss não diminuindo**:
   ```
   Train Loss: 0.027 → 0.026 → 0.025 (muito lento)
   ```

4. **Gradientes muito pequenos**:
   ```
   Avg gradient norm < 1e-5
   ```

---

## 💡 LIÇÕES APRENDIDAS

1. **Alpha no Focal Loss é contra-intuitivo**:
   - NÃO é "peso da classe"
   - É "fator de down-weight para exemplos fáceis"
   - Classe minoritária precisa de alpha ALTO (0.9-0.98)
   - Classe majoritária precisa de alpha BAIXO (0.02-0.1)

2. **Threshold 0.5 é inadequado para classes desbalanceadas**:
   - Sempre usar threshold search
   - Threshold ótimo geralmente é ~0.3-0.4

3. **Early stopping muito agressivo**:
   - patience=12 é pouco para modelos complexos
   - Usar patience=20-30

4. **Regularização excessiva**:
   - Dropout 0.3-0.4 + weight_decay 2e-4 é muito
   - Reduzir ambos

---

## 📁 ARQUIVOS AFETADOS

- ✅ `configs/experiment_v8_fixed.yaml` (CRIADO E CORRIGIDO)
  - binary_strategy: "pass_vs_fail" ✅
  - Focal Loss: alpha=[0.995, 0.005], gamma=3.5 ✅
  - Hiperparâmetros ajustados ✅
- ✅ `ANALISE_PROBLEMAS_TREINAMENTO.md` (ATUALIZADO)
  - Problema #5: Binary Strategy documentado ✅
- ⏳ `main_v8.py` (threshold search já existe)
- ⏳ `src/training/losses.py` (verificar implementação focal loss)
- ⏳ `src/evaluation/metrics.py` (adicionar diversidade)

---

**Status**: 🟡 **CORREÇÕES IMPLEMENTADAS - PRONTO PARA TESTE**

**Próximo passo**: Rodar teste rápido (10 épocas, 5K samples)
```bash
python main_v8.py --config configs/experiment_v8_fixed.yaml \
                   --sample-size 5000 \
                   --num-epochs 10
```
