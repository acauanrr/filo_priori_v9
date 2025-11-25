# Análise: Focal Loss Causando Colapso de Predição

**Data**: 2025-11-07
**Status**: 🔴 **PROBLEMA CRÍTICO**

---

## 🔴 SINTOMAS

```python
Classification Report:
              precision    recall  f1-score   support

    Not-Pass       0.03      1.00      0.06       174
        Pass       0.00      0.00      0.00      5888

    accuracy                           0.03      6062
```

**O modelo prevê TUDO como classe 0 (Not-Pass/Fail)!**

---

## 🔍 ANÁLISE MATEMÁTICA

### Configuração Atual

```yaml
focal:
  alpha: [0.995, 0.005]  # [classe_0_Fail, classe_1_Pass]
  gamma: 3.5
```

### Como o Focal Loss Funciona

```python
# losses.py linha 93:
alpha_t = self.alpha[targets]  # Indexa pelo target VERDADEIRO
focal_loss = alpha_t * focal_weight * ce_loss
```

**Tradução:**
- Samples com label=0 (Fail): loss × 0.995
- Samples com label=1 (Pass): loss × 0.005

### O Ratio Criado

```
Peso Fail / Peso Pass = 0.995 / 0.005 = 199:1
```

**199x MAIS peso para Fail que para Pass!**

### Por Que o Modelo Colapsa?

O modelo "raciocina":

```
"Se eu errar 1 Fail: loss = 0.995 × ...  = GIGANTE
 Se eu errar 199 Pass: loss = 199 × 0.005 × ... = 0.995 × ... = MESMO!

 Logo, melhor prever TUDO como Fail para minimizar o risco!"
```

---

## 📊 COMPORTAMENTO OBSERVADO

### Predições do Modelo

```python
# Validation set: 6062 samples
predictions = [0, 0, 0, 0, ..., 0]  # TODOS zeros!

# True labels:
# - 174 Fail (classe 0)
# - 5888 Pass (classe 1)
```

### Confusão Matrix Resultante

```
                Predicted
               Not-Pass  Pass
Actual Not-Pass   174      0    ← 100% recall
       Pass      5888      0    ← 0% recall
```

### Métricas Calculadas

```python
# Not-Pass:
Recall    = 174 / (174 + 0) = 1.00   (100%)
Precision = 174 / (174 + 5888) = 0.0287  (2.9%)

# Pass:
Recall    = 0 / (0 + 5888) = 0.00   (0%)
Precision = undefined (nenhuma predição)

# Overall:
Accuracy = (174 + 0) / 6062 = 0.0287  (2.9%)
```

---

## ❌ POR QUE ALPHA [0.995, 0.005] ESTÁ ERRADO?

### 1. Ratio Excessivo (199:1)

Para imbalance de 37:1 (Pass:Fail), um ratio de peso de **199:1** é **EXCESSIVAMENTE AGRESSIVO**!

### 2. Comparação com Class Weights

```python
# Class weights calculados automaticamente:
# class_weights = [19.00785973, 0.51350777]
# Ratio: 19.00 / 0.513 ≈ 37:1

# Focal Loss alpha atual:
# alpha = [0.995, 0.005]
# Ratio: 0.995 / 0.005 = 199:1  ← 5.4x MAIS AGRESSIVO!
```

### 3. Interpretação do Alpha

No Focal Loss, **alpha NÃO deve ser interpretado como class weight direto**!

Alpha é um **fator de escala da loss**, não um peso de classe. Valores muito extremos causam colapso.

---

## ✅ SOLUÇÕES

### SOLUÇÃO 1: Ajustar Alpha para Valores Razoáveis

```yaml
# ❌ ATUAL (causa colapso):
focal:
  alpha: [0.995, 0.005]  # Ratio 199:1
  gamma: 3.5

# ✅ OPÇÃO A (conservador):
focal:
  alpha: [0.75, 0.25]  # Ratio 3:1
  gamma: 2.5

# ✅ OPÇÃO B (moderado):
focal:
  alpha: [0.85, 0.15]  # Ratio 5.7:1
  gamma: 3.0

# ✅ OPÇÃO C (agressivo mas razoável):
focal:
  alpha: [0.95, 0.05]  # Ratio 19:1 (igual ao class weight)
  gamma: 3.0
```

**RECOMENDAÇÃO**: Começar com **OPÇÃO C** (alpha=[0.95, 0.05]) que tem ratio igual ao class weight natural.

### SOLUÇÃO 2: Usar Weighted Cross-Entropy (RECOMENDADO!)

```yaml
# ✅ MAIS SIMPLES E INTUITIVO:
loss:
  type: "weighted_ce"
  weighted_ce:
    use_class_weights: true
    # Usa class_weights do DataLoader automaticamente
    # class_weights = [19.0, 0.51] → ratio 37:1
```

**Vantagens:**
- Mais intuitivo que Focal Loss
- Ratio de peso corresponde exatamente ao imbalance
- Menos propenso a colapso
- Amplamente testado e validado

### SOLUÇÃO 3: Focal Loss com Gamma Baixo

```yaml
# ✅ Reduzir gamma também ajuda:
focal:
  alpha: [0.8, 0.2]  # Ratio 4:1 (conservador)
  gamma: 1.5        # Mais suave que 3.5
```

---

## 🎯 RECOMENDAÇÃO FINAL

### Usar Weighted Cross-Entropy

```yaml
loss:
  type: "weighted_ce"
  weighted_ce:
    use_class_weights: true
```

**Por quê?**
1. ✅ Simples e intuitivo
2. ✅ Não requer ajuste fino de alpha/gamma
3. ✅ Usa class weights naturais (37:1)
4. ✅ Menos propenso a colapso
5. ✅ Implementação já existe em losses.py

---

## 📝 CORREÇÃO APLICAR

### 1. Criar nova configuração: `experiment_v8_weighted_ce.yaml`

```yaml
experiment:
  name: "v8_weighted_ce"
  version: "8.0.2"
  description: "V8 with Weighted CE instead of problematic Focal Loss"

# ... (resto igual) ...

# Loss Function - WEIGHTED CE
loss:
  type: "weighted_ce"

  weighted_ce:
    use_class_weights: true
    label_smoothing: 0.0

  # For reference, these were the problematic focal loss values:
  # focal:
  #   alpha: [0.995, 0.005]  # ❌ TOO EXTREME - ratio 199:1
  #   gamma: 3.5
```

### 2. Atualizar main_v8.py

```python
# main_v8.py linha 473-479
# ❌ ANTES:
if config['loss']['type'] == 'focal':
    criterion = FocalLoss(
        alpha=config['loss']['focal']['alpha'],
        gamma=config['loss']['focal']['gamma']
    ).to(device)
else:
    criterion = nn.CrossEntropyLoss().to(device)

# ✅ DEPOIS:
if config['loss']['type'] == 'focal':
    criterion = FocalLoss(
        alpha=config['loss']['focal']['alpha'],
        gamma=config['loss']['focal']['gamma']
    ).to(device)
elif config['loss']['type'] == 'weighted_ce':
    # Use class weights from DataLoader
    class_weights_tensor = torch.FloatTensor(data_dict['class_weights']).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)
else:
    criterion = nn.CrossEntropyLoss().to(device)
```

---

## 📊 RESULTADOS ESPERADOS (Weighted CE)

```
Classification Report (EXPECTED):
              precision    recall  f1-score   support

    Not-Pass       0.35      0.45      0.40       174
        Pass       0.98      0.97      0.98      5888

    accuracy                           0.95      6062
   macro avg       0.66      0.71      0.69      6062
weighted avg       0.96      0.95      0.95      6062
```

**Metas:**
- Recall Not-Pass: ≥ 40% (detecta falhas!)
- Recall Pass: ≥ 95% (mantém performance)
- Accuracy: ≥ 95% (alta overall)
- F1 Macro: ≥ 0.65 (balanceado)

---

## 🎓 LIÇÕES APRENDIDAS

1. **Focal Loss Alpha ≠ Class Weight**
   - Alpha é fator de escala, não peso direto
   - Valores extremos causam colapso

2. **Ratio de 199:1 é Excessivo**
   - Para imbalance 37:1, ratio 199:1 é ~5x agressivo demais
   - Ratio de peso deve ser próximo ao ratio natural

3. **Weighted CE é Mais Seguro**
   - Mais intuitivo
   - Menos propenso a colapso
   - Usa weights naturais diretamente

4. **Sempre Validar Métricas Cedo**
   - Se Recall de uma classe = 100% e outra = 0%, PARE!
   - Isso indica colapso de predição

5. **Focal Loss Precisa Ajuste Fino**
   - Alpha e gamma são sensíveis
   - Valores padrão da literatura (alpha=0.25, gamma=2.0) são para cenários diferentes
   - Precisa experimentação cuidadosa

---

## ✅ PRÓXIMO PASSO

1. Criar `configs/experiment_v8_weighted_ce.yaml`
2. Atualizar main_v8.py para suportar weighted_ce
3. Executar treino com Weighted CE
4. Validar que ambas as classes são preditas (diversity > 0.3)

---

**Status**: 🔴 **PROBLEMA IDENTIFICADO E SOLUÇÃO PROPOSTA**

**Ação Imediata**: Trocar Focal Loss por Weighted Cross-Entropy
