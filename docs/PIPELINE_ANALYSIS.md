# Análise Completa da Pipeline - Filo-Priori

**Data:** 2024-11-14  
**Status:** ⚠️ PROBLEMAS IDENTIFICADOS - CORREÇÕES NECESSÁRIAS

---

## Resumo Executivo

**Problema Principal:** O `main.py` está **quebrado** após a organização. Ele importa encoders antigos (Qodo) que foram **deletados**.

**Impacto:** 
- ❌ Pipeline não vai executar
- ❌ Imports falhando
- ❌ Dimensões incorretas em alguns lugares

**Solução:** Atualizar `main.py` para usar o novo sistema SBERT com `EmbeddingManager`.

---

## Análise Detalhada por Componente

### 1. ❌ Embeddings (QUEBRADO)

**Problema:**
```python
# main.py linhas 42-43 (QUEBRADO!)
from embeddings.qodo_encoder import QodoEncoder  # ❌ ARQUIVO NÃO EXISTE
from embeddings.qodo_encoder_chunked import QodoEncoderChunked  # ❌ DELETADO
```

**Uso atual:**
```python
# Linhas 175-182
if use_chunked:
    encoder = QodoEncoderChunked(semantic_config, device='cuda')  # ❌ QUEBRA AQUI
else:
    encoder = QodoEncoder(semantic_config, device='cuda')  # ❌ OU AQUI
```

**Solução:**
```python
# CORRETO:
from embeddings import EmbeddingManager

# Uso:
manager = EmbeddingManager(config, force_regenerate=args.force_regen)
embeddings = manager.get_embeddings(train_df, test_df)

# Acesso:
train_tc_emb = embeddings['train_tc']  # (N, 768)
train_commit_emb = embeddings['train_commit']  # (N, 768)
```

**Status:** ❌ **CRÍTICO - Pipeline quebrada**

---

### 2. ✅ Features Estruturais (OK)

**Análise:**
```python
# main.py linhas 230-267
extractor = StructuralFeatureExtractor(...)
train_struct = extractor.transform(df_train, is_test=False)
# Output: (N, 6) - correto!
```

**Features extraídas:**
1. Pass rate (histórico)
2. Fail rate (histórico)
3. Recent pass rate (janela recente)
4. Recent fail rate (janela recente)
5. Dias desde último teste
6. Total de execuções

**Imputação:**
```python
# Linhas 269-304
# Usa similarity semântica para imputation - OK!
val_struct, stats = impute_structural_features(
    train_embeddings, train_struct, tc_keys_train,
    val_embeddings, val_struct, tc_keys_val,
    extractor.tc_history,
    k_neighbors=10
)
```

**Status:** ✅ **OK - Funcionando**

---

### 3. ✅ Grafo Filogenético (OK)

**Análise:**
```python
# main.py linhas 373-388
graph_builder = build_phylogenetic_graph(
    df_train,
    graph_type=graph_config['type'],  # 'co_failure'
    min_co_occurrences=graph_config['min_co_occurrences'],  # 2
    weight_threshold=graph_config['weight_threshold'],  # 0.1
    cache_path=graph_cache_path
)
```

**Output:**
```python
# Linhas 414-422
edge_index, edge_weights = graph_builder.get_edge_index_and_weights(
    tc_keys=all_tc_keys,
    return_torch=True
)
# edge_index: [2, num_edges]
# edge_weights: [num_edges]
```

**Mapping:**
```python
# Linhas 424-437
tc_key_to_global_idx = {tc_key: idx for idx, tc_key in enumerate(all_tc_keys)}
train_data['global_indices'] = np.array([tc_key_to_global_idx[tc_key] 
                                          for tc_key in df_train['TC_Key']])
```

**Status:** ✅ **OK - Funcionando**

---

### 4. ⚠️ Modelo Dual-Stream (ATENÇÃO)

**Config atual:**
```yaml
# configs/experiment.yaml (NOVO)
model:
  semantic:
    input_dim: 1536  # TC (768) + Commit (768) ✅ CORRETO
    hidden_dim: 256
    num_layers: 2
    
  structural:
    input_dim: 6  # ✅ CORRETO
    hidden_dim: 64
    num_layers: 2
    
  gnn:
    type: "GAT"
    hidden_dim: 128
    num_layers: 2
    num_heads: 4
```

**Código do modelo:**
```python
# src/models/dual_stream_v8.py linha 649
def create_model_v8(config: Dict) -> DualStreamModelV8:
    # Lê input_dim do config - OK!
    semantic_stream = SemanticStream(
        input_dim=config['semantic']['input_dim'],  # 1536 ✅
        hidden_dim=config['semantic']['hidden_dim'],  # 256
        ...
    )
```

**Forward pass esperado:**
```python
# Entrada:
semantic_emb: [batch, 1536]  # TC (768) + Commit (768)
structural_feat: [batch, 6]
edge_index: [2, num_edges]
edge_weights: [num_edges]
global_indices: [batch]

# Processamento:
# 1. Semantic Stream: [batch, 1536] → [batch, 256]
# 2. Structural Stream: [batch, 6] → [batch, 64]
# 3. GAT Stream: extrai subgraph → [batch, 128]
# 4. Fusion: concat → [batch, 448] → [batch, 256]
# 5. Classifier: [batch, 256] → [batch, 2]
```

**Status:** ⚠️ **ATENÇÃO - Config atualizado, mas main.py precisa correção**

---

### 5. ✅ Training Loop (OK - após correção)

**Estrutura:**
```python
# main.py linha 481
def train_epoch(model, loader, criterion, optimizer, device, 
                edge_index, edge_weights, all_structural_features, num_nodes_global):
    for batch in loader:
        semantic_emb, structural_feat, labels, global_indices = batch
        
        # Forward
        outputs = model(
            semantic_emb.to(device),
            structural_feat.to(device),
            edge_index,
            edge_weights,
            global_indices.to(device),
            all_structural_features.to(device),
            num_nodes_global
        )
        
        # Loss
        loss = criterion(outputs, labels.to(device))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Status:** ✅ **OK - Lógica correta**

---

### 6. ✅ Evaluation & Metrics (OK)

**Métricas computadas:**
```python
# main.py linha 558
def evaluate(...):
    # Coleta predições
    all_outputs = []
    all_labels = []
    all_probs = []
    
    # Calcula métricas
    metrics = compute_metrics(
        y_true=all_labels,
        y_pred=predictions,
        y_prob=all_probs
    )
    
    # Retorna:
    # - accuracy
    # - precision
    # - recall
    # - f1_macro
    # - f1_weighted
    # - auprc_macro
    # - auroc
```

**Status:** ✅ **OK - Métricas completas**

---

### 7. ✅ Ranking & APFD (OK)

**Processo:**
```python
# main.py linhas 849-932
# STEP 5: APFD no test set do modelo

# 1. Coleta predições
df_test['prediction'] = predictions
df_test['fail_probability'] = fail_probs

# 2. Ranking por probabilidade (decrescente)
df_ranked = df_test.sort_values('fail_probability', ascending=False)

# 3. Calcula APFD por build
apfd_results_df, apfd_summary = generate_apfd_report(
    df=df_ranked,
    build_column='Build',
    fault_column='TE_Test_Result',  # ✅ USA COLUNA CORRETA
    fault_value='Fail',
    output_path=apfd_path
)

# 4. Output:
#   - apfd_per_build.csv
#   - Mean APFD calculado
```

**APFD no test.csv completo:**
```python
# main.py linhas 939-1095
# STEP 6: Processa test.csv COMPLETO (277 builds)

# 1. Carrega test.csv
df_full_test = pd.read_csv('datasets/test.csv')

# 2. Gera embeddings
full_test_emb = ...

# 3. Predições
full_test_probs = model.predict(...)

# 4. Ranking
df_full_ranked = df_full_test.sort_values('fail_probability', ascending=False)

# 5. APFD por build
apfd_results_df_full, apfd_summary_full = generate_apfd_report(
    df=df_full_ranked,
    build_column='Build',
    fault_column='TE_Test_Result',
    fault_value='Fail',
    output_path='apfd_per_build_FULL_testcsv.csv'
)

# 6. Output:
#   - apfd_per_build_FULL_testcsv.csv
#   - prioritized_test_cases_FULL_testcsv.csv
#   - Mean APFD across 277 builds
```

**Status:** ✅ **OK - Lógica de APFD correta**

---

## Problemas Identificados

### Crítico ❌

1. **main.py importa encoders deletados**
   - Linha 42: `from embeddings.qodo_encoder import QodoEncoder`
   - Linha 43: `from embeddings.qodo_encoder_chunked import QodoEncoderChunked`
   - **Impacto:** Pipeline não vai executar
   - **Solução:** Substituir por `EmbeddingManager`

### Importante ⚠️

2. **main.py não tem flag --force-regen-embeddings**
   - O script `run_experiment.sh` passa essa flag
   - Mas `main.py` não a reconhece
   - **Impacto:** Flag é ignorada
   - **Solução:** Adicionar argumento ao parser

3. **Encoder retornado em prepare_data() não é usado**
   - Linha 438: `return ..., encoder, ...`
   - Mas encoder é do tipo Qodo (que não existe)
   - **Impacto:** Variável não utilizada, mas quebra se tentar usar
   - **Solução:** Remover do return ou retornar EmbeddingManager

### Menor ℹ️

4. **Comentários desatualizados**
   - Vários comentários mencionam Qodo e dimensão 3072
   - **Impacto:** Confusão na documentação
   - **Solução:** Atualizar comentários

---

## Arquivos de Saída Gerados

### ✅ Corretos

1. **Métricas:**
   - `test_metrics.json` - Todas as métricas de classificação
   - `train_history.json` - Histórico de training

2. **Predições:**
   - `predictions.csv` - Predições do modelo no test set

3. **Rankings:**
   - `prioritized_test_cases.csv` - Ranking do test set
   - `prioritized_test_cases_FULL_testcsv.csv` - Ranking completo (277 builds)

4. **APFD:**
   - `apfd_per_build.csv` - APFD por build (test set)
   - `apfd_per_build_FULL_testcsv.csv` - APFD completo (277 builds)
   - **Mean APFD** incluído em ambos

5. **Visualizações:**
   - `confusion_matrix.png`
   - `precision_recall_curves.png`

6. **Modelo:**
   - `best_model.pt` - Melhor checkpoint
   - `config_used.yaml` - Config do experimento

---

## Fluxo Correto da Pipeline (Após Correção)

```
1. EMBEDDINGS (CORRIGIDO)
   ├─ EmbeddingManager inicializado
   ├─ Verifica cache
   ├─ Carrega ou gera embeddings
   │  ├─ Train: TC (768) + Commit (768) = 1536
   │  ├─ Val: TC (768) + Commit (768) = 1536
   │  └─ Test: TC (768) + Commit (768) = 1536
   └─ Salva em cache (se novo)

2. FEATURES ESTRUTURAIS ✅
   ├─ Extração de 6 features por TC
   ├─ Imputação para TCs sem histórico
   └─ Output: (N, 6)

3. GRAFO FILOGENÉTICO ✅
   ├─ Construção do grafo de co-failure
   ├─ Edge index: [2, num_edges]
   ├─ Edge weights: [num_edges]
   └─ Mapping TC_Key → global_index

4. DATA LOADERS ✅
   ├─ Train: (embeddings, structural, labels, global_indices)
   ├─ Val: idem
   └─ Test: idem

5. MODELO ✅
   ├─ Semantic Stream: [batch, 1536] → [batch, 256]
   ├─ Structural Stream: [batch, 6] → [batch, 64]
   ├─ GAT Stream: subgraph → [batch, 128]
   ├─ Fusion: [batch, 448] → [batch, 256]
   └─ Classifier: [batch, 256] → [batch, 2]

6. TRAINING ✅
   ├─ Forward pass
   ├─ Loss (Focal ou Weighted CE)
   ├─ Backward
   ├─ Optimizer step
   └─ Early stopping

7. EVALUATION ✅
   ├─ Predições no test set
   ├─ Métricas de classificação
   ├─ Threshold optimization
   └─ Salva métricas

8. RANKING & APFD ✅
   ├─ Ranking por fail_probability
   ├─ APFD por build (test set)
   ├─ APFD no test.csv completo (277 builds)
   ├─ Mean APFD calculado
   └─ CSVs salvos

9. OUTPUT FILES ✅
   ├─ test_metrics.json
   ├─ predictions.csv
   ├─ prioritized_test_cases.csv
   ├─ prioritized_test_cases_FULL_testcsv.csv
   ├─ apfd_per_build.csv
   ├─ apfd_per_build_FULL_testcsv.csv
   ├─ confusion_matrix.png
   ├─ precision_recall_curves.png
   ├─ best_model.pt
   └─ config_used.yaml
```

---

## Checklist de Correções Necessárias

### Críticas (Impedem Execução)

- [ ] **main.py**: Remover imports de Qodo encoders
- [ ] **main.py**: Adicionar import de EmbeddingManager
- [ ] **main.py**: Substituir código de embedding (linhas 169-228)
- [ ] **main.py**: Adicionar flag --force-regen-embeddings ao parser
- [ ] **main.py**: Atualizar função prepare_data() para usar EmbeddingManager

### Importantes (Melhorias)

- [ ] **main.py**: Atualizar comentários (Qodo → SBERT, 3072 → 1536)
- [ ] **main.py**: Remover variável `encoder` do return (ou retornar manager)
- [ ] Criar arquivo de validação da pipeline
- [ ] Testar execução end-to-end

### Opcionais (Futuro)

- [ ] Adicionar progress bars mais detalhadas
- [ ] Adicionar checkpoint intermediário
- [ ] Melhorar logging de APFD

---

## Tempo Estimado de Correção

- **Correções críticas:** 30-45 minutos
- **Testes de validação:** 15-30 minutos
- **Total:** ~1 hora

---

## Próximos Passos

1. **Corrigir main.py** para usar EmbeddingManager
2. **Testar execução** com dados pequenos (sample_size=100)
3. **Validar outputs** (verificar CSVs gerados)
4. **Executar experimento completo**
5. **Documentar** resultados

---

**Status Final:** ⚠️ **PIPELINE QUEBRADA - CORREÇÕES NECESSÁRIAS**

**Prioridade:** 🔴 **ALTA - Bloqueia uso do sistema**

---

*Análise completa em 2024-11-14*
