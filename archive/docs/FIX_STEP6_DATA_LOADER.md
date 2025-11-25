# Fix: STEP 6 Data Loader Scope Issue

**Data**: 2025-11-07
**Status**: ✅ **CORRIGIDO**

---

## 🔴 PROBLEMA IDENTIFICADO

Durante a execução do `experiment_v8_weighted_ce`, o STEP 6 (processamento do test.csv completo) falhou:

```
ERROR:__main__:
❌ ERROR processing full test.csv: name 'data_loader' is not defined
ERROR:__main__:   Continuing with split test results only...
Traceback (most recent call last):
  File "/home/acauanribeiro/iats/filo_priori_v8/main_v8.py", line 652, in main
    test_df_full = data_loader.load_full_test_dataset()
                   ^^^^^^^^^^^
NameError: name 'data_loader' is not defined. Did you mean: 'DataLoader'?
```

**Consequência**: Processou apenas 64 builds do split de teste ao invés dos 277 builds esperados do test.csv completo.

---

## 🔍 CAUSA RAIZ

O objeto `data_loader` (e outros como `encoder`, `text_processor`, `extractor`) foram criados dentro da função `prepare_data()` mas **não foram retornados**. Quando o STEP 6 tentou usá-los, estavam fora de escopo.

```python
# prepare_data() linha 84
def prepare_data(config: Dict, sample_size: int = None):
    data_loader = DataLoader(config)  # Criado aqui
    encoder = SemanticEncoder(...)
    text_processor = TextProcessor()
    extractor = StructuralFeatureExtractor(...)
    # ... código ...
    return train_data, val_data, test_data, ...  # ❌ Não retornava data_loader!

# main() linha 652
def main(args):
    # ... código ...
    test_df_full = data_loader.load_full_test_dataset()  # ❌ Erro: data_loader não existe!
```

---

## ✅ SOLUÇÃO IMPLEMENTADA

### 1. Atualizar `prepare_data()` para Retornar Objetos Necessários

**Linha 272** (antes linha 266):
```python
return (train_data, val_data, test_data, graph_builder, edge_index, edge_weights,
        class_weights, data_loader, encoder, text_processor, extractor)
        #            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #            ADICIONADO: 4 objetos necessários para STEP 6
```

### 2. Atualizar Docstring

**Linhas 75-77**:
```python
Returns:
    Tuple of (train_data, val_data, test_data, graph_builder, edge_index, edge_weights,
              class_weights, data_loader, encoder, text_processor, extractor)
```

### 3. Atualizar `main()` para Receber Objetos

**Linhas 452-453**:
```python
(train_data, val_data, test_data, graph_builder, edge_index, edge_weights,
 class_weights, data_loader, encoder, text_processor, extractor) = prepare_data(config, args.sample_size)
```

---

## 🚀 IMPACTO DA CORREÇÃO

### Antes (com erro)
```
STEP 6: PROCESSING FULL TEST.CSV FOR FINAL APFD
❌ ERROR: name 'data_loader' is not defined
⚠️  WARNING: Expected 277 builds but got 64
```

### Depois (esperado)
```
STEP 6: PROCESSING FULL TEST.CSV FOR FINAL APFD
6.1: Loading FULL test.csv...
✅ Loaded full test.csv:
   Total samples: ~180K
   Total builds: ~1000+
   Builds with 'Fail': 277

6.2: Generating semantic embeddings for full test set...
✅ Generated embeddings: (180K, 1024)

6.3: Extracting structural features for full test set...
✅ Extracted features: (180K, 6)
✅ Imputed 50123 unknown tests

6.4: Generating predictions on full test set...
✅ Generated predictions: 180K samples

6.5: Generating prioritized CSV (FULL test.csv)...
✅ Saved: prioritized_test_cases_FULL_testcsv.csv

6.6: Calculating APFD per build (FULL test.csv)...
✅ SUCCESS: Found exactly 277 builds with failures!
📊 Mean APFD: 0.XXXX (across 277 builds)  ← MÉTRICA FINAL!
```

---

## 📝 VALIDAÇÃO

### 1. Verificar Sintaxe
```bash
python -m py_compile main_v8.py
# ✅ Sem erros de sintaxe
```

### 2. Executar Novamente
```bash
python main_v8.py --config configs/experiment_v8_weighted_ce.yaml --device cuda
```

### 3. Verificar Resultados
```bash
# Deve ter 277 builds (278 linhas com header)
wc -l results/experiment_v8_weighted_ce/apfd_per_build_FULL_testcsv.csv

# Deve mostrar 277 builds
grep "total_builds" results/experiment_v8_weighted_ce/tmux-buffer.txt

# Deve ter ~180K linhas (+ header)
wc -l results/experiment_v8_weighted_ce/prioritized_test_cases_FULL_testcsv.csv
```

---

## 🎯 PRÓXIMA AÇÃO

### Executar Novamente com Correção

```bash
# O modelo JÁ ESTÁ TREINADO! (best_model.pt existe)
# Mas precisa re-rodar STEP 6 para processar test.csv completo

python main_v8.py --config configs/experiment_v8_weighted_ce.yaml --device cuda
```

**Nota**: Se quiser APENAS re-rodar STEP 6 sem re-treinar:
1. O código vai carregar automaticamente `best_model.pt` se existir
2. Pode pular epochs já treinadas se implementar checkpoint loading
3. Ou criar script separado para processar apenas test.csv

---

## 📊 RESULTADOS ESPERADOS

Com a correção, o STEP 6 deve processar com sucesso:

| Métrica | Valor Esperado |
|---------|----------------|
| Total test.csv samples | ~180,000 |
| Total builds | ~1000+ |
| **Builds com Fail** | **277** ✅ |
| Prioritized CSV | ✅ Criado |
| APFD per build | ✅ 277 builds |
| Mean APFD (final) | 0.55-0.60 |

---

## ✅ STATUS

- [x] Problema identificado
- [x] Causa raiz analisada
- [x] Solução implementada
- [x] Sintaxe validada
- [ ] Re-execução para validação final

**Próxima Ação**: Executar `python main_v8.py --config configs/experiment_v8_weighted_ce.yaml --device cuda`

---

**Correção completa em**: `main_v8.py` (linhas 76-77, 272, 452-453)
