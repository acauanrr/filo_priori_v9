# 🔴 DIAGNÓSTICO CRÍTICO: Colapso Persistente

## 📊 RESUMO DOS EXPERIMENTOS

| Experimento | Focal Alpha | Focal Gamma | Sampling Ratio | Class Weights | Resultado |
|-------------|-------------|-------------|----------------|---------------|-----------|
| **Exp 01** (baseline) | 0.25 | 2.0 | Nenhum | Não | ❌ Colapso → Pass |
| **Exp 02** (improved) | 0.75 | 3.0 | 20:1 | Sim (19:1) | ❌ Colapso → Fail |
| **Exp 03** (balanced) | 0.25 | 2.0 | 5:1 | Sim (19:1) | ❌ Colapso → Fail |

---

## 🔍 ANÁLISE EXPERIMENTO 03

### Configuração Aplicada
```yaml
Loss: WeightedFocalLoss
  - focal_alpha: 0.25 (reduzido de 0.75)
  - focal_gamma: 2.0 (reduzido de 3.0)
  - class_weights: [19.13, 0.51] (automático)

Sampling: BalancedSampling
  - minority_weight: 1.0
  - majority_weight: 0.2 (5:1 ratio, reduzido de 20:1)
  - Expected minority: 11.83% (reduzido de 35%)
```

### Resultados: AINDA COLAPSADO! ❌

```
TREINAMENTO (Épocas 1-13):
  Val Accuracy: 0.0283 (SEMPRE IGUAL!)
  Val F1 Macro: 0.0275 (SEMPRE IGUAL!)
  Early stopping: Epoch 13

TESTE:
              precision  recall  f1-score  support
    Not-Pass      0.03    1.00      0.05      157  ← Prediz TUDO Fail
        Pass      0.00    0.00      0.00     5995  ← NUNCA prediz Pass

  Test F1 Macro: 0.0249  (pior que baseline!)
  Test Accuracy: 2.55%   (pior que baseline!)
```

### 🔴 PROBLEMA IDENTIFICADO

**QUALQUER combinação de Focal + Weights + Sampling causa colapso!**

Análise matemática do weight total (Exp 03):
```python
# Configuração Exp 03
alpha = 0.25
gamma = 2.0
class_weights = [19.13, 0.51]
sampling = 5:1

# Para p=0.5 (incerteza):
minority_weight = 0.25 * (1-0.5)^2.0 * 19.13 = 1.20
majority_weight = 0.75 * (1-0.5)^2.0 * 0.51 = 0.10
Ratio = 1.20 / 0.10 = 12:1

# Com sampling (modelo vê 11.83% minority vs 2.6% real):
Effective oversampling = 4.5x
Total effective weight = 12 * 4.5 = 54:1 !!!
```

**CONCLUSÃO**: Mesmo "reduzido", o peso ainda é **54x** maior para minority!

---

## 💡 SOLUÇÃO: ABORDAGEM CONSERVADORA PASSO A PASSO

### Estratégia: Testar técnicas ISOLADAMENTE

Precisamos descobrir qual técnica funciona SOZINHA antes de combinar.

### 🎯 EXPERIMENTO 04a: WEIGHTED CE APENAS (RECOMENDADO)

**Configuração**:
```yaml
Loss: Weighted Cross-Entropy
  - type: "weighted_ce"
  - class_weights: [19.13, 0.51] (automático)
  - NO Focal Loss
  - NO Balanced Sampling

Sampling: Normal (sem balanceamento)
  - Shuffle padrão
  - Dataset real: 2.6% Fail, 97.4% Pass
```

**Vantagens**:
- ✅ Simples e estável
- ✅ Apenas 1 mecanismo de rebalanceamento
- ✅ Amplamente usado e testado
- ✅ Menos hiper-parâmetros

**Expectativa**:
- F1 Macro: 0.30-0.40
- Recall Not-Pass: 0.20-0.40
- Recall Pass: 0.95-0.98
- Sem colapso (ambas classes preditas)

---

### 🎯 EXPERIMENTO 04b: FOCAL LOSS APENAS (ALTERNATIVA)

**Configuração**:
```yaml
Loss: Focal Loss
  - type: "focal"
  - focal_alpha: 0.5 (moderado)
  - focal_gamma: 2.0
  - use_class_weights: false  ← SEM class weights
  - NO Balanced Sampling

Sampling: Normal (sem balanceamento)
```

**Vantagens**:
- ✅ Focal cuida do imbalance sozinho
- ✅ Foco em hard examples
- ✅ Não depende de class weights

**Expectativa**:
- F1 Macro: 0.25-0.35
- Pode precisar ajuste de alpha

---

### 🎯 EXPERIMENTO 04c: BASELINE PURO (CONTROLE)

**Configuração**:
```yaml
Loss: Cross-Entropy padrão
  - type: "ce"
  - NO weights
  - NO Focal
  - NO Sampling

Sampling: Normal
```

**Objetivo**:
- Baseline para comparar se técnicas ajudam
- Pode colapsar para Pass (como Exp 01)
- Mas estabelece floor de performance

---

## 📋 PLANO DE AÇÃO

### Passo 1: Executar Exp 04a (Weighted CE)
```bash
./venv/bin/python main.py --config configs/experiment_04a_weighted_ce_only.yaml
```

**Critérios de Sucesso**:
- [ ] Ambas classes preditas (diversity > 30%)
- [ ] F1 Macro > 0.30
- [ ] Recall Pass > 0.95
- [ ] Recall Not-Pass > 0.20
- [ ] Nenhum colapso

**Se FALHAR**: Ir para Exp 04c (baseline puro)

### Passo 2: SE Exp 04a funcionar
Tentar adicionar **1 técnica por vez**:
- Exp 05a: Weighted CE + Threshold Optimization
- Exp 05b: Weighted CE + Sampling LEVE (2:1)
- Exp 05c: Weighted CE + Focal LEVE (alpha=0.1)

### Passo 3: SE Exp 04a falhar
Ir para abordagens alternativas:
- Data augmentation (SMOTE)
- Cost-sensitive learning diferente
- Ensemble methods
- Two-stage training

---

## 🔧 OUTRAS POSSÍVEIS CAUSAS DO COLAPSO

### 1. Modelo muito complexo para dados esparsos
```
GAT com 2 layers + 4 heads = muitos parâmetros
Dataset: 1,323 Fails (minority)
Ratio parâmetros/samples muito alto
```

**Solução**: Simplificar modelo (1 GAT layer, 2 heads)

### 2. Learning rate muito alto
```
Atual: 5e-5
Para imbalance extremo: 1e-5 pode ser melhor
```

**Solução**: Reduzir LR

### 3. Dropout muito alto
```
Atual: 0.15-0.3
Com poucos samples minority: dropout pode destruir sinais fracos
```

**Solução**: Reduzir dropout para 0.1

### 4. Graph MUITO denso interferindo
```
Density: 21.36% (588K edges)
GAT pode estar propagando ruído demais
```

**Solução**: Reduzir semantic_top_k de 10 para 5

---

## 🎯 CONFIGURAÇÃO EXPERIMENTO 04a (PRONTO PARA USO)

```yaml
# configs/experiment_04a_weighted_ce_only.yaml

experiment:
  name: "experiment_04a_weighted_ce_only"
  description: "Weighted CE apenas - sem Focal, sem Sampling"

training:
  loss:
    type: "weighted_ce"
    label_smoothing: 0.0
    # Class weights automáticos

  sampling:
    use_balanced_sampling: false  # ← SEM sampling!

  learning_rate: 0.00003  # ↓ Reduzido de 5e-5

model:
  # Simplificado
  gnn:
    num_layers: 1  # ↓ de 2
    num_heads: 2   # ↓ de 4
    dropout: 0.1   # ↓ de 0.2

  classifier:
    dropout: 0.2   # ↓ de 0.3

graph:
  use_multi_edge: true
  semantic_top_k: 5  # ↓ de 10 (menos denso)
```

---

## 📊 EXPECTATIVAS REALISTAS

### Imbalance 37:1 é EXTREMO

Literatura mostra que para ratios > 20:1:
- F1 Macro: 0.30-0.50 é EXCELENTE
- F1 Minority: 0.20-0.40 é BOM
- Recall Minority: 0.30-0.60 é ACEITÁVEL

**Nossas metas anteriores (F1=0.50-0.55) podem ser OTIMISTAS demais!**

### Metas Revisadas (Realistas)

**Mínimo Aceitável**:
- F1 Macro: 0.30
- Recall Not-Pass: 0.25
- Recall Pass: 0.95
- APFD: 0.55

**Alvo**:
- F1 Macro: 0.40
- Recall Not-Pass: 0.35
- Recall Pass: 0.97
- APFD: 0.60

**Excelente**:
- F1 Macro: 0.50
- Recall Not-Pass: 0.50
- Recall Pass: 0.98
- APFD: 0.65

---

## 🚨 SE TUDO FALHAR

### Abordagens Alternativas

1. **Two-Stage Training**:
   - Stage 1: Treinar para detectar qualquer failure (alta recall)
   - Stage 2: Fine-tune para melhorar precision

2. **Threshold Otimizado + Ensemble**:
   - Treinar 3-5 modelos com seeds diferentes
   - Ensemble voting
   - Threshold otimizado por build

3. **SMOTE Agressivo**:
   - Oversampling até 1:1 ratio
   - Criar samples sintéticos de Fail
   - Pode overfittar mas vale testar

4. **Metric Learning**:
   - Triplet Loss
   - Aprender embedding space onde Fails são separados

5. **Aceitar Limitação**:
   - Imbalance 37:1 pode ser INTRATÁVEL
   - Focar em APFD (que já está razoável)
   - Aceitar que ranking > classificação

---

## ✅ PRÓXIMA AÇÃO IMEDIATA

**CRIAR E EXECUTAR EXPERIMENTO 04a**

```bash
# 1. Criar config simplificado
# (já vou criar nos próximos comandos)

# 2. Executar
./venv/bin/python main.py --config configs/experiment_04a_weighted_ce_only.yaml

# 3. Monitorar:
# - Ambas classes sendo preditas?
# - F1 > 0.30?
# - Métricas variando entre épocas?
```

**Tempo estimado**: 2-3 horas

**Se funcionar**: Gradualmente adicionar técnicas
**Se falhar**: Tentar Exp 04b ou 04c

---

## 📝 LIÇÕES APRENDIDAS

1. **Mais não é sempre melhor**: Combinar TODAS as técnicas causou colapso inverso

2. **Imbalance extremo é difícil**: 37:1 ratio está no limite do tratável

3. **Teste isolado é crítico**: Precisamos saber qual técnica funciona sozinha

4. **Multi-Edge Graph funciona**: Densidade 21% é sucesso, manter!

5. **Paciência e método**: Abordagem científica iterativa é necessária

---

**Conclusão**: O colapso persiste porque estamos usando MÚLTIPLAS técnicas de rebalanceamento que se MULTIPLICAM em vez de somar. A solução é **simplificar drasticamente** e testar técnicas **isoladamente**.
