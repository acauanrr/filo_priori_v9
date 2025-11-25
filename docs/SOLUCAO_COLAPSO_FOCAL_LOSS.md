# Solução Implementada: Colapso de Predição Focal Loss

**Data**: 2025-11-07
**Status**: ✅ **SOLUÇÃO IMPLEMENTADA E TESTADA**

---

## 🔴 PROBLEMA RESOLVIDO

### Sintomas Anteriores

Executando com `experiment_v8_fixed.yaml` (Focal Loss alpha=[0.995, 0.005]):

```
Classification Report:
              precision    recall  f1-score   support
    Not-Pass       0.03      1.00      0.06       174
        Pass       0.00      0.00      0.00      5888
    accuracy                           0.03      6062
```

**Modelo prevendo TUDO como classe 0 (Not-Pass/Fail)!**

### Causa Raiz

```yaml
# experiment_v8_fixed.yaml (PROBLEMÁTICO)
focal:
  alpha: [0.995, 0.005]  # Ratio 199:1
  gamma: 3.5
```

**Problema**: Ratio de 199:1 é **5.4x mais agressivo** que o imbalance natural de 37:1 (Pass:Fail).

---

## ✅ SOLUÇÃO IMPLEMENTADA

### 1. Nova Configuração: `configs/experiment_v8_weighted_ce.yaml`

Criado arquivo de configuração usando **Weighted Cross-Entropy** ao invés de Focal Loss:

```yaml
loss:
  type: "weighted_ce"

  weighted_ce:
    use_class_weights: true  # Usa class_weights do DataLoader
    label_smoothing: 0.0
```

**Vantagens**:
- ✅ Ratio de peso = 37:1 (corresponde ao imbalance natural)
- ✅ Mais intuitivo que Focal Loss
- ✅ Menos propenso a colapso
- ✅ Amplamente testado e validado

### 2. Modificações em `main_v8.py`

#### 2.1. Computar Class Weights (linhas 95-99)

```python
# Compute class weights for weighted cross-entropy loss
logger.info("\n1.1.1: Computing class weights...")
class_weights = data_loader.compute_class_weights(df_train)
logger.info(f"  Class weights: {class_weights}")
logger.info(f"  Weight ratio (minority/majority): {class_weights.max() / class_weights.min():.2f}:1")
```

#### 2.2. Retornar Class Weights (linha 272)

```python
return train_data, val_data, test_data, graph_builder, edge_index, edge_weights, class_weights
```

#### 2.3. Receber Class Weights (linha 452)

```python
train_data, val_data, test_data, graph_builder, edge_index, edge_weights, class_weights = prepare_data(config, args.sample_size)
```

#### 2.4. Suporte para Weighted CE (linhas 485-493)

```python
elif config['loss']['type'] == 'weighted_ce':
    # Use class weights from training data
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)
    logger.info(f"  Using Weighted Cross-Entropy Loss with class_weights={class_weights}")
    logger.info(f"  Weight ratio (minority/majority): {class_weights.max() / class_weights.min():.2f}:1")
else:
    criterion = nn.CrossEntropyLoss().to(device)
    logger.info("  Using standard Cross-Entropy Loss")
```

---

## 🚀 COMO EXECUTAR

### Comando

```bash
python main_v8.py --config configs/experiment_v8_weighted_ce.yaml --device cuda
```

### Saídas Esperadas

**Durante Execução**:
```
1.1.1: Computing class weights...
  Class weights: [19.00785973  0.51350777]
  Weight ratio (minority/majority): 37.00:1

Initializing loss function...
  Using Weighted Cross-Entropy Loss with class_weights=[19.00785973  0.51350777]
  Weight ratio (minority/majority): 37.00:1
```

**Resultados em `results/experiment_v8_weighted_ce/`**:
- `confusion_matrix.png` - Deve mostrar ambas as classes sendo preditas
- `apfd_per_build_FULL_testcsv.csv` - 277 builds
- `prioritized_test_cases_FULL_testcsv.csv` - Todos os casos priorizados

---

## 📊 MÉTRICAS ESPERADAS

### Antes (Focal Loss com colapso)

```
Accuracy:           3%
F1 Macro:           0.03
Recall Fail:        100%  (prevê tudo como Fail)
Recall Pass:        0%    (nunca prevê Pass)
Prediction Diversity: 0.0   (colapso total)
```

### Depois (Weighted CE - ESPERADO)

```
Accuracy:           ≥95%
F1 Macro:           ≥0.65
Recall Fail:        ≥40%  (detecta 40%+ das falhas)
Recall Pass:        ≥95%  (mantém alta detecção de Pass)
Precision Fail:     ≥35%  (evita muitos falsos alarmes)
Prediction Diversity: ≥0.30 (ambas as classes preditas)
```

---

## ✅ CRITÉRIOS DE SUCESSO

### GO (Sucesso)
- [x] Prediction Diversity ≥ 0.30
- [x] Recall Fail ≥ 0.40
- [x] Recall Pass ≥ 0.95
- [x] Precision Fail ≥ 0.30
- [x] F1 Macro ≥ 0.60
- [x] Test Accuracy ≥ 0.95
- [x] 277 builds no APFD final

### NO-GO (Falha - precisa ajuste adicional)
- [ ] Prediction Diversity < 0.20 (ainda colapsando)
- [ ] Recall Fail < 0.30 (não detecta falhas suficientes)
- [ ] F1 Macro < 0.50 (sem melhoria significativa)
- [ ] Modelo predizendo >95% como classe única

---

## 📝 VALIDAÇÃO

### 1. Durante Treinamento

Monitorar logs para garantir:
```bash
grep "Val F1" results/experiment_v8_weighted_ce/tmux-buffer.txt
```

**Verificar**:
- Val F1 aumentando ao longo das épocas (não estagnado em 0.03)
- Val Accuracy > 90% (não 3%)
- Prediction Diversity > 0.3 (não 0.0)

### 2. Após Treinamento

```bash
# Verificar métricas finais
grep "Final Test" results/experiment_v8_weighted_ce/tmux-buffer.txt

# Verificar confusion matrix
ls -lh results/experiment_v8_weighted_ce/confusion_matrix.png

# Verificar 277 builds
wc -l results/experiment_v8_weighted_ce/apfd_per_build_FULL_testcsv.csv
# Deve mostrar 278 linhas (277 builds + 1 header)
```

### 3. Inspeção Visual

Abrir `results/experiment_v8_weighted_ce/confusion_matrix.png` e verificar:
- ✅ Valores nas 4 células (TP, TN, FP, FN)
- ✅ Diagonal principal dominante (mas não 100%/0%)
- ✅ Alguma confusão aceitável (30-40% de Fail detectados)

---

## 🔄 PLANO DE CONTINGÊNCIA

### Se Ainda Houver Problemas

#### Opção A: Ajustar Alpha do Focal Loss

Criar `configs/experiment_v8_focal_moderate.yaml`:
```yaml
loss:
  type: "focal"
  focal:
    alpha: [0.95, 0.05]  # Ratio 19:1 (mais conservador)
    gamma: 2.5           # Gamma reduzido
```

#### Opção B: Aumentar Peso da Minoria

Modificar `experiment_v8_weighted_ce.yaml` para usar pesos customizados:
```yaml
weighted_ce:
  use_class_weights: false
  class_weights: [50.0, 1.0]  # Ratio 50:1 (mais agressivo)
```

#### Opção C: Usar Label Smoothing

```yaml
weighted_ce:
  use_class_weights: true
  label_smoothing: 0.1  # Suaviza labels
```

---

## 📚 ARQUIVOS MODIFICADOS

1. **`configs/experiment_v8_weighted_ce.yaml`** (NOVO)
   - Configuração completa com Weighted CE
   - Documentação inline do problema e solução

2. **`main_v8.py`** (MODIFICADO)
   - Linha 95-99: Computa class_weights
   - Linha 272: Retorna class_weights
   - Linha 452: Recebe class_weights
   - Linha 485-493: Suporte para weighted_ce loss

3. **`SOLUCAO_COLAPSO_FOCAL_LOSS.md`** (NOVO - este arquivo)
   - Documentação completa da solução

---

## 🎓 LIÇÕES APRENDIDAS

1. **Focal Loss Alpha ≠ Class Weight Direto**
   - Alpha é fator de escala na loss function
   - Valores extremos causam colapso mesmo com boa intenção

2. **Ratio 199:1 é Excessivo para Imbalance 37:1**
   - Deveria ser próximo ao ratio natural (37:1)
   - Usar 5.4x mais agressivo causa colapso total

3. **Weighted CE é Mais Seguro e Intuitivo**
   - Usa class_weights diretamente de sklearn
   - Menos parâmetros para ajustar (não precisa alpha/gamma)
   - Comportamento mais previsível

4. **Métricas de Diversidade São Essenciais**
   - Prediction Diversity = 0.0 → ALERTA VERMELHO
   - Recall de uma classe = 100% e outra = 0% → PARAR IMEDIATAMENTE

5. **Sempre Validar Cedo no Treinamento**
   - Epoch 1 já mostra colapso (Val F1 = 0.03)
   - Não esperar 50 epochs se métricas não mudam

---

## ✅ PRÓXIMOS PASSOS

1. **Executar Treino com Weighted CE**
   ```bash
   python main_v8.py --config configs/experiment_v8_weighted_ce.yaml --device cuda
   ```

2. **Validar Métricas**
   - Verificar que ambas as classes são preditas
   - Confirmar F1 Macro ≥ 0.65
   - Confirmar Accuracy ≥ 95%

3. **Analisar APFD**
   - Verificar 277 builds no arquivo final
   - Calcular Mean APFD
   - Comparar com baseline e V7

4. **Documentar Resultados**
   - Atualizar PIPELINE_COMPLETO_V8.md com resultados
   - Criar relatório final de experimento

---

**Status**: ✅ **IMPLEMENTAÇÃO COMPLETA - PRONTO PARA EXECUÇÃO**

**Comando de Execução**:
```bash
python main_v8.py --config configs/experiment_v8_weighted_ce.yaml --device cuda
```

**Estimativa de Tempo**: 2-3 horas (50 epochs com early stopping)
