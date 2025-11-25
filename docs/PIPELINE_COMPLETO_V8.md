# Pipeline Completo Filo-Priori V8

**Data**: 2025-11-07
**Status**: ✅ **PRONTO PARA EXECUÇÃO**

---

## 📋 VISÃO GERAL

Pipeline completo do Filo-Priori V8 com todas as correções, imputações e processamento do test.csv completo implementado.

---

## 🚀 EXECUÇÃO

```bash
python main_v8.py --config configs/experiment_v8_fixed.yaml --device cuda
```

**Duração estimada**: 2-3 horas (treino completo + processamento test.csv)

---

## 📊 PIPELINE COMPLETO (6 STEPS)

### STEP 1: DATA PREPARATION

```
1.1: Loading datasets (train.csv)
     ├── Split: 80% train / 10% val / 10% test
     ├── Strategy: pass_vs_fail (APENAS Pass vs Fail)
     └── Samples excluídos: Delete, Blocked, Conditional Pass, etc.

1.2: Generating semantic embeddings (BGE)
     ├── Model: models/finetuned_bge_v1
     ├── Dim: 1024
     ├── Fields: TE_Summary + TC_Steps + commit
     └── Caching: cache/embeddings/

1.3: Extracting structural features
     ├── Features: [test_age, failure_rate, recent_failure_rate,
     │              flakiness_rate, commit_count, test_novelty]
     ├── Fitted on training data
     ├── Global statistics computed (means, medians, stds)
     └── Caching: cache/structural_features.pkl

1.3b: Imputing missing structural features
      ├── Method 1: Semantic Similarity (k=10, threshold=0.5)
      ├── Method 2: Fallback conservador (médias populacionais)
      ├── Validation: ~77% imputação semântica, ~23% fallback
      └── Adds Gaussian noise to avoid identical features

1.4: Building phylogenetic graph
     ├── Type: co_failure
     ├── Min co-occurrences: 2
     ├── Nodes: test cases
     ├── Edges: co-failure relationships
     └── Caching: cache/phylogenetic_graph.pkl

1.5: Extracting graph structure
     └── edge_index, edge_weights for GAT
```

### STEP 2: MODEL CREATION

```
2.1: Creating Dual-Stream V8 Model
     ├── Semantic Stream: MLP (1024 → 256)
     ├── Structural Stream: GAT (6 → 256)
     ├── Fusion: Cross-Attention (4 heads)
     ├── Classifier: MLP (512 → 128 → 64 → 2)
     └── Total params: ~2.0M
```

### STEP 3: TRAINING

```
3.1: Loss Function
     ├── Type: Focal Loss
     ├── Alpha: [0.995, 0.005]  # CRITICAL: invertido e agressivo!
     ├── Gamma: 3.5
     └── Strategy: Penaliza classe majoritária (Pass), foca em Fail

3.2: Training Loop
     ├── Epochs: 50
     ├── Batch size: 32
     ├── Optimizer: AdamW (lr=1e-4, wd=5e-5)
     ├── Scheduler: CosineAnnealingLR
     ├── Early Stopping: patience=20, monitor=val_f1_macro
     └── Best model saved: best_model_v8.pt

3.3: Validation per epoch
     └── Metrics: accuracy, f1_macro, f1_weighted, precision, recall, auprc
```

### STEP 4: TEST EVALUATION (Split)

```
4.1: Load best model
4.2: Evaluate on test split (~10% of train.csv)
4.3: Compute metrics
     └── Classification report, confusion matrix, PR curves
```

### STEP 5: APFD CALCULATION (Split)

```
5.1: Add probabilities to test DataFrame
     ├── probability = P(Fail) = probs[:, 0]
     └── CRITICAL: Uses TE_Test_Result column for correct labels

5.2: Verify columns
     ├── TE_Test_Result: must exist! (original labels)
     ├── Build_ID: must exist! (for per-build APFD)
     └── Validation logs shown

5.3: Generate prioritized CSV
     ├── File: results/experiment_v8_fixed/prioritized_test_cases.csv
     ├── Columns: Build_ID, TC_Key, TE_Test_Result, label_binary,
     │            probability, diversity_score, priority_score, rank
     └── Ranks: per-build, 1-indexed (1 = highest priority)

5.4: Calculate APFD per build
     ├── File: results/experiment_v8_fixed/apfd_per_build.csv
     ├── Only builds with at least 1 Fail
     ├── Business rule: builds with 1 TC → APFD=1.0
     └── Expected: ~10-30 builds (test split is only 10%)

5.5: Print APFD summary
     └── Mean, median, std, min, max, distribution
```

### STEP 6: PROCESS FULL TEST.CSV (277 BUILDS) ⭐ NOVO!

```
6.1: Load FULL test.csv
     ├── File: datasets/test.csv
     ├── Total samples: ~31,333
     ├── Total builds: ~1,000+
     └── Builds with Fail: 277 (expected)

6.2: Generate semantic embeddings for full test
     ├── Uses same BGE model
     ├── No caching (one-time processing)
     └── Shape: [31333, 1024]

6.3: Extract structural features for full test
     ├── Uses fitted extractor from training
     ├── is_test=True (uses historical stats only)
     └── Shape: [31333, 6]

6.3b: Impute missing features
      ├── Identifies which samples need imputation
      ├── Uses semantic similarity with training samples
      └── Fallback to conservative defaults

6.4: Generate predictions on full test
     ├── Batch processing (batch_size=32)
     ├── Uses best trained model
     ├── Structural stream: GAT on full graph once
     └── Output: [31333, 2] probabilities

6.5: Prepare data for APFD
     ├── probability = P(Fail) = probs[:, 0]
     ├── label_binary from TE_Test_Result == 'Fail'
     └── Verify counts: failures vs passes

6.6: Generate prioritized CSV (FULL)
     ├── File: results/experiment_v8_fixed/prioritized_test_cases_FULL_testcsv.csv
     ├── All 31,333 test cases with ranks per build
     └── Format: same as split version

6.7: Calculate APFD per build (FULL)
     ├── File: results/experiment_v8_fixed/apfd_per_build_FULL_testcsv.csv
     ├── Expected: EXACTLY 277 builds
     ├── Each row: method_name, build_id, test_scenario, count_tc,
     │             count_commits, apfd, time
     └── Mean APFD: PRIMARY METRIC!

6.8: Validation
     ├── Check: total_builds == 277
     ├── SUCCESS if 277, WARNING otherwise
     └── Log all file paths
```

---

## 📁 ARQUIVOS DE SAÍDA

### Estrutura de Resultados

```
results/experiment_v8_fixed/
├── best_model_v8.pt                              # Melhor modelo treinado
├── config_used.yaml                               # Configuração usada
├── confusion_matrix.png                           # Matriz de confusão (split test)
├── precision_recall_curves.png                    # Curvas PR (split test)
├── prioritized_test_cases.csv                     # Test split prioritizado
├── apfd_per_build.csv                            # APFD do test split (~30 builds)
├── prioritized_test_cases_FULL_testcsv.csv       # ⭐ FULL test.csv prioritizado (31K)
└── apfd_per_build_FULL_testcsv.csv               # ⭐ APFD dos 277 builds (PRINCIPAL!)
```

### Descrição dos Arquivos Principais

#### 1. `apfd_per_build_FULL_testcsv.csv` ⭐ PRINCIPAL

**Formato**:
```csv
method_name,build_id,test_scenario,count_tc,count_commits,apfd,time
v8_fixed_FULL_testcsv,Build_001,full_test_csv_277_builds,45,12,0.6234,0
v8_fixed_FULL_testcsv,Build_002,full_test_csv_277_builds,38,8,0.7891,0
...
```

**Estatísticas**:
- Total de linhas: **EXATAMENTE 277**
- Colunas:
  - `method_name`: Nome do experimento + "_FULL_testcsv"
  - `build_id`: Identificador do build
  - `test_scenario`: "full_test_csv_277_builds"
  - `count_tc`: Número de TCs únicos neste build
  - `count_commits`: Número de commits únicos (incluindo CRs)
  - `apfd`: APFD score [0, 1] (higher is better)
  - `time`: Placeholder (0)

**Métrica Principal**: `Mean APFD` (média da coluna `apfd`)

#### 2. `prioritized_test_cases_FULL_testcsv.csv`

**Formato**:
```csv
Build_ID,TC_Key,TE_Test_Result,label_binary,probability,diversity_score,priority_score,rank
Build_001,TC_12345,Fail,1,0.8234,0.0,0.8234,1
Build_001,TC_67890,Pass,0,0.7123,0.0,0.7123,2
Build_001,TC_11111,Pass,0,0.5432,0.0,0.5432,3
...
```

**Total de linhas**: ~31,333 (todos os TCs do test.csv)

**Colunas**:
- `Build_ID`: Build onde o TC foi executado
- `TC_Key`: Identificador único do test case
- `TE_Test_Result`: Resultado original ("Pass", "Fail")
- `label_binary`: 1 se Fail, 0 se Pass
- `probability`: P(Fail) predita pelo modelo
- `diversity_score`: Sempre 0.0 (não usado em V8)
- `priority_score`: Mesmo que probability
- `rank`: Prioridade no build (1 = mais alta, N = mais baixa)

---

## ✅ VALIDAÇÕES AUTOMÁTICAS

Durante a execução, o pipeline valida:

### 1. Colunas Críticas

```
✅ TE_Test_Result column found with 2 unique values
   Values: {'Pass': 29679, 'Fail': 1654}
✅ Build_ID column found: 1234 unique builds
```

### 2. Imputação de Features

```
  Validation samples needing imputation: 0/6917 (0.0%)
  Test samples needing imputation: 127/8127 (1.6%)

  Imputation complete:
    Semantic-based: 98 (77.2%)
    Fallback (conservative): 29 (22.8%)
```

### 3. APFD - 277 Builds

```
FINAL APFD RESULTS - FULL TEST.CSV (277 BUILDS)
======================================================================
Total builds analyzed: 277
Mean APFD: 0.XXXX ⭐ PRIMARY METRIC

VALIDATION
======================================================================
✅ SUCCESS: Found exactly 277 builds with failures!
✅ Mean APFD: 0.XXXX
```

**Se não encontrar 277**:
```
⚠️  WARNING: Expected 277 builds but found XXX
   This may indicate incorrect filtering or data issues
```

---

## 🎯 CRITÉRIOS DE SUCESSO

### Métricas de Classificação (Test Split)

| Métrica | GO (Sucesso) | REVIEW | NO-GO |
|---------|--------------|--------|-------|
| **Prediction Diversity** | ≥ 0.20 | [0.15, 0.20) | < 0.15 |
| **Recall Fail** | ≥ 0.30 | [0.20, 0.30) | < 0.20 |
| **Precision Fail** | ≥ 0.25 | [0.20, 0.25) | < 0.20 |
| **F1 Macro** | ≥ 0.50 | [0.45, 0.50) | < 0.45 |
| **Test Accuracy** | ≥ 0.80 | [0.75, 0.80) | < 0.75 |

### Métricas de APFD (FULL test.csv) ⭐ PRINCIPAL

| Métrica | Target | Minimum | Notes |
|---------|--------|---------|-------|
| **Total Builds** | 277 | 277 | **MUST BE EXACT** |
| **Mean APFD** | ≥ 0.60 | ≥ 0.55 | Higher is better |
| **Median APFD** | ≥ 0.65 | ≥ 0.60 | Less affected by outliers |
| **Builds APFD ≥ 0.7** | ≥ 50% | ≥ 40% | Good prioritization |
| **Builds APFD < 0.5** | < 20% | < 30% | Random or worse |

---

## 🔍 COMO VERIFICAR RESULTADOS

### 1. Verificar Execução Bem-Sucedida

```bash
# Verificar se todos os arquivos foram criados
ls -lh results/experiment_v8_fixed/

# Deve mostrar:
# - best_model_v8.pt
# - apfd_per_build_FULL_testcsv.csv  <-- PRINCIPAL
# - prioritized_test_cases_FULL_testcsv.csv
```

### 2. Validar 277 Builds

```bash
# Contar linhas no arquivo APFD (deve ser 278 = 277 builds + 1 header)
wc -l results/experiment_v8_fixed/apfd_per_build_FULL_testcsv.csv

# Deve mostrar: 278 results/experiment_v8_fixed/apfd_per_build_FULL_testcsv.csv
```

### 3. Calcular Mean APFD

```bash
# Calcular média da coluna APFD (coluna 6)
awk -F',' 'NR>1 {sum+=$6; count++} END {print "Mean APFD:", sum/count; print "Total Builds:", count}' \
    results/experiment_v8_fixed/apfd_per_build_FULL_testcsv.csv
```

**Saída esperada**:
```
Mean APFD: 0.XXXX
Total Builds: 277
```

### 4. Verificar Distribuição de APFD

```bash
# Contar builds por faixa de APFD
awk -F',' 'NR>1 {
    apfd=$6;
    if (apfd >= 0.7) high++;
    else if (apfd >= 0.5) medium++;
    else low++;
}
END {
    print "APFD >= 0.7:", high;
    print "0.5 <= APFD < 0.7:", medium;
    print "APFD < 0.5:", low;
}' results/experiment_v8_fixed/apfd_per_build_FULL_testcsv.csv
```

---

## 📊 INTERPRETAÇÃO DE RESULTADOS

### Mean APFD

```
APFD = 1.0:  Perfeito - todas as falhas detectadas primeiro
APFD ≥ 0.7:  Excelente - maioria das falhas detectadas cedo
APFD ≥ 0.6:  Bom - performance acima da média
APFD ≥ 0.55: Aceitável - melhor que random (0.5)
APFD ≥ 0.5:  Limiar - igual a random
APFD < 0.5:  Ruim - pior que random
```

### Exemplo de Log de Sucesso

```
======================================================================
FINAL APFD RESULTS - FULL TEST.CSV (277 BUILDS)
======================================================================
Total builds analyzed: 277
Total test cases: 31333
Mean TCs per build: 113.1

APFD Statistics:
  Mean:   0.6234 ⭐ PRIMARY METRIC
  Median: 0.6578
  Std:    0.1234
  Min:    0.2345
  Max:    0.9876

APFD Distribution:
  Builds with APFD = 1.0:   12 (  4.3%)
  Builds with APFD ≥ 0.7:  145 ( 52.3%)
  Builds with APFD ≥ 0.5:  231 ( 83.4%)
  Builds with APFD < 0.5:   46 ( 16.6%)
======================================================================

VALIDATION
======================================================================
✅ SUCCESS: Found exactly 277 builds with failures!
✅ Mean APFD: 0.6234

✅ All results saved to: results/experiment_v8_fixed/
   - prioritized_test_cases.csv (test split)
   - apfd_per_build.csv (test split)
   - prioritized_test_cases_FULL_testcsv.csv (all 277 builds)
   - apfd_per_build_FULL_testcsv.csv (all 277 builds)
======================================================================

TRAINING COMPLETE!
======================================================================
Best Val F1: 0.5678
Test F1: 0.5432
Mean APFD (test split): 0.5891
Mean APFD (FULL test.csv, 277 builds): 0.6234 ⭐
======================================================================
```

---

## 🐛 TROUBLESHOOTING

### Problema 1: Não encontrou 277 builds

```
⚠️  WARNING: Expected 277 builds but found XXX
```

**Possíveis causas**:
1. `binary_strategy` não é "pass_vs_fail" → Verifica config
2. test.csv está incompleto → Verifica arquivo
3. Filtros muito agressivos no data_loader → Verifica _clean_data_non_strict

**Solução**:
```bash
# Verificar test.csv diretamente
python -c "
import pandas as pd
df = pd.read_csv('datasets/test.csv')
print(f'Total samples: {len(df)}')
print(f'Total builds: {df[\"Build_ID\"].nunique()}')
builds_fail = df[df['TE_Test_Result'] == 'Fail']['Build_ID'].nunique()
print(f'Builds with Fail: {builds_fail}')
"
```

### Problema 2: TE_Test_Result não encontrado

```
❌ CRITICAL: TE_Test_Result column not found in test DataFrame!
```

**Causa**: DataLoader está dropando a coluna

**Solução**: Verificar que `load_full_test_dataset()` está preservando todas as colunas.

### Problema 3: Out of Memory durante Step 6

**Causa**: Test.csv completo é muito grande para processar de uma vez

**Solução**:
```python
# Em main_v8.py, aumentar batch_size ou processar em chunks
test_loader_full = torch.utils.data.DataLoader(
    test_dataset_full,
    batch_size=16,  # Reduzir de 32 para 16
    shuffle=False
)
```

---

## 📝 CHECKLIST DE VALIDAÇÃO

Após executar o pipeline:

- [ ] Arquivo `best_model_v8.pt` criado
- [ ] Arquivo `apfd_per_build_FULL_testcsv.csv` criado
- [ ] Arquivo tem EXATAMENTE 278 linhas (277 + header)
- [ ] Mean APFD ≥ 0.55
- [ ] Log mostra "✅ SUCCESS: Found exactly 277 builds"
- [ ] Todos os 4 arquivos CSV criados em results/
- [ ] Não há erros ou warnings críticos no log
- [ ] Test F1 Macro ≥ 0.50
- [ ] Recall Fail ≥ 0.30
- [ ] Prediction Diversity ≥ 0.20

---

## 🎓 SUMÁRIO

✅ **Pipeline Completo Implementado**:
- STEP 1-5: Treino e avaliação no split test
- STEP 6: **Processamento FULL test.csv (277 builds)** ⭐ NOVO!

✅ **Arquivos Principais Criados**:
- `apfd_per_build_FULL_testcsv.csv` (277 builds)
- `prioritized_test_cases_FULL_testcsv.csv` (31K test cases)

✅ **Validações Automáticas**:
- Verifica 277 builds
- Valida colunas críticas
- Confirma imputação de features

✅ **Métrica Principal**: **Mean APFD** dos 277 builds

---

**Status**: ✅ **PRONTO PARA EXECUÇÃO**

**Comando**:
```bash
python main_v8.py --config configs/experiment_v8_fixed.yaml --device cuda
```

**Duração**: 2-3 horas

**Resultado esperado**: Mean APFD ≥ 0.55 nos 277 builds do test.csv completo! 🚀
