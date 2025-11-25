# Problemas na Integração APFD do V8

**Data**: 2025-11-07
**Status**: 🔴 **CORREÇÕES NECESSÁRIAS**

---

## 📋 RESUMO EXECUTIVO

A implementação do módulo APFD (`src/evaluation/apfd.py`) está **CORRETA** e completa, seguindo as regras de negócio:
1. ✅ Calcula APFD por build
2. ✅ Filtra apenas builds com pelo menos 1 falha (esperado: 277 builds)
3. ✅ Regra especial: builds com 1 TC têm APFD=1.0
4. ✅ Conta commits totais (incluindo CRs)

**PORÉM**, a **integração no main_v8.py** tem **4 problemas críticos** que impedem o cálculo correto do APFD.

---

## 🔴 PROBLEMA #1: Path Hardcoded (Não Usa output_dir)

### Código Atual (main_v8.py:508)

```python
apfd_results, apfd_summary = generate_apfd_report(
    test_df,
    method_name=config['experiment']['name'],
    test_scenario="v8_full_test",
    output_path="apfd_per_build_v8.csv"  # ❌ HARDCODED!
)
```

### Problema
- Path hardcoded não usa `config['output']['results_dir']`
- Arquivo é salvo na raiz do projeto, não na pasta do experimento
- Dificulta organização de múltiplos experimentos

### Solução (Como no V7:716-727)

```python
# Usar output_dir da configuração
results_dir = config['output']['results_dir']
apfd_path = os.path.join(results_dir, 'apfd_per_build.csv')

apfd_results, apfd_summary = generate_apfd_report(
    test_df_with_ranks,  # Note: com ranks!
    method_name=config['experiment']['name'],
    test_scenario="v8_full_test",
    output_path=apfd_path
)
```

---

## 🔴 PROBLEMA #2: label_binary Incorreto

### Código Atual (main_v8.py:501)

```python
test_df['probability'] = test_probs[:, 0]  # P(Fail) ✅ CORRETO
test_df['label_binary'] = test_data['labels']  # ❌ INCORRETO!
```

### Problema
`test_data['labels']` vem do processamento do DataLoader que:
- **Com pass_vs_all**: Mistura Fail + Delete + Blocked como "Not-Pass" (classe 0)
- **Com pass_vs_fail**: Exclui Delete/Blocked do dataset (mas ainda pode ter problemas)

**Regra de negócio**: APFD deve usar **APENAS** `TE_Test_Result == 'Fail'` do CSV original!

### Por que isso é Crítico?

Exemplo:
- Build tem 10 TCs: 8 Pass, 1 Fail, 1 Delete
- **Incorreto**: label_binary conta Delete como "falha" → APFD errado
- **Correto**: Apenas Fail conta como falha → APFD correto

### Solução (Como no V7:696-701)

```python
test_df['probability'] = test_probs[:, 0]  # P(Fail)

# CRITICAL: Use TE_Test_Result from original CSV for APFD
if 'TE_Test_Result' in test_df.columns:
    test_df['label_binary'] = (test_df['TE_Test_Result'].astype(str).str.strip() == 'Fail').astype(int)
else:
    logger.error("❌ CRITICAL: TE_Test_Result column not found!")
    logger.error("   APFD calculation will be incorrect!")
    # Fallback (not ideal)
    test_df['label_binary'] = test_data['labels']
```

---

## 🔴 PROBLEMA #3: Falta generate_prioritized_csv

### Código Atual
V8 chama diretamente `generate_apfd_report` sem gerar o CSV prioritizado.

### Problema
- Não salva `prioritized_test_cases.csv` com ranks
- Não permite análise posterior das predições
- Não permite replicação dos resultados

### Solução (Como no V7:716-724)

```python
# Generate prioritized CSV with ranks per build
prioritized_path = os.path.join(results_dir, 'prioritized_test_cases.csv')
test_df_with_ranks = generate_prioritized_csv(
    test_df,
    output_path=prioritized_path,
    probability_col='probability',
    label_col='label_binary',
    build_col='Build_ID'
)
logger.info(f"✅ Prioritized test cases saved to: {prioritized_path}")

# THEN calculate APFD per build
apfd_path = os.path.join(results_dir, 'apfd_per_build.csv')
apfd_results_df, apfd_summary = generate_apfd_report(
    test_df_with_ranks,  # ✅ Use df with ranks
    method_name=config['experiment']['name'],
    test_scenario="v8_full_test",
    output_path=apfd_path
)
```

---

## 🔴 PROBLEMA #4: Não Verifica TE_Test_Result

### Código Atual
V8 não verifica se a coluna `TE_Test_Result` existe no DataFrame.

### Problema
- `TE_Test_Result` é **ESSENCIAL** para APFD correto
- Se a coluna não existir, APFD será calculado errado silenciosamente
- Dificulta debugging

### Solução (Como no V7:703-713)

```python
# CRITICAL: Verify TE_Test_Result exists (should come from original CSV)
if 'TE_Test_Result' not in test_df.columns:
    logger.error("❌ CRITICAL: TE_Test_Result column not found in test DataFrame!")
    logger.error("   This column is required for correct APFD calculation.")
    logger.error("   APFD should only count builds with TE_Test_Result == 'Fail'")
    logger.error("   Cannot proceed with APFD calculation.")
    return
else:
    logger.info(f"✅ TE_Test_Result column found with {len(test_df['TE_Test_Result'].unique())} unique values")
    logger.info(f"   Values: {test_df['TE_Test_Result'].value_counts().to_dict()}")
```

---

## 🔴 PROBLEMA #5: Probability Column Invertida (POSSÍVEL)

### Código Atual (main_v8.py:500)

```python
test_df['probability'] = test_probs[:, 0]  # P(Fail)
```

### Análise

**Com pass_vs_fail strategy**:
- Classe 0 = Fail (minority class)
- Classe 1 = Pass (majority class)
- `test_probs[:, 0]` = P(Fail) ✅ **CORRETO para APFD!**

**PORÉM**, se o modelo retorna softmax:
- `test_probs[:, 0]` = probabilidade da classe 0 (Fail)
- Para APFD, queremos **maior probabilidade = maior prioridade**
- Então `test_probs[:, 0]` é correto!

**Status**: ✅ **PROVAVELMENTE CORRETO** (precisa validação após treino)

---

## 📊 COMPARAÇÃO: V7 vs V8

| Aspecto | V7 (✅ Correto) | V8 (❌ Problemas) |
|---------|----------------|-------------------|
| **Output Path** | `os.path.join(output_dir, 'apfd_per_build.csv')` | `"apfd_per_build_v8.csv"` (hardcoded) |
| **label_binary** | `(test_df['TE_Test_Result'] == 'Fail').astype(int)` | `test_data['labels']` (incorreto) |
| **Verifica TE_Test_Result** | ✅ Sim (lines 703-713) | ❌ Não |
| **Gera prioritized_test_cases.csv** | ✅ Sim (line 717) | ❌ Não |
| **Probability Column** | `failure_probs` (correto) | `test_probs[:, 0]` (provavelmente correto) |
| **Adiciona ranks** | ✅ Via `generate_prioritized_csv` | ⚠️ Via `generate_apfd_report` (interno) |

---

## 🔧 CORREÇÃO COMPLETA

### Substituir linhas 493-519 do main_v8.py por:

```python
# APFD calculation
logger.info("\n"+"="*70)
logger.info("STEP 5: APFD CALCULATION")
logger.info("="*70)

# Add probabilities to test DataFrame
test_df = test_data['df'].copy()
test_df['probability'] = test_probs[:, 0]  # P(Fail) - class 0 with pass_vs_fail

# CRITICAL: Use TE_Test_Result from original CSV for correct APFD
if 'TE_Test_Result' not in test_df.columns:
    logger.error("❌ CRITICAL: TE_Test_Result column not found in test DataFrame!")
    logger.error("   This column is required for correct APFD calculation.")
    logger.error("   APFD should only count builds with TE_Test_Result == 'Fail'")
    logger.error("   Check if data_loader is preserving this column from test.csv")
    # Fallback: create from pass_vs_fail labels (not ideal but better than nothing)
    logger.warning("   Using fallback: mapping labels to TE_Test_Result")
    test_df['TE_Test_Result'] = test_data['labels'].map({0: 'Fail', 1: 'Pass'})
else:
    logger.info(f"✅ TE_Test_Result column found with {len(test_df['TE_Test_Result'].unique())} unique values")
    logger.info(f"   Values: {test_df['TE_Test_Result'].value_counts().to_dict()}")

# Create label_binary from TE_Test_Result (not from processed labels)
test_df['label_binary'] = (test_df['TE_Test_Result'].astype(str).str.strip() == 'Fail').astype(int)
logger.info(f"   label_binary distribution: {test_df['label_binary'].value_counts().to_dict()}")

# Verify Build_ID exists
if 'Build_ID' not in test_df.columns:
    logger.error("❌ CRITICAL: Build_ID column not found!")
    logger.error("   Cannot calculate APFD per build.")
else:
    logger.info(f"✅ Build_ID column found: {test_df['Build_ID'].nunique()} unique builds")

# Get results directory from config
results_dir = config['output']['results_dir']
os.makedirs(results_dir, exist_ok=True)

# Generate prioritized CSV with ranks per build
prioritized_path = os.path.join(results_dir, 'prioritized_test_cases.csv')
test_df_with_ranks = generate_prioritized_csv(
    test_df,
    output_path=prioritized_path,
    probability_col='probability',
    label_col='label_binary',
    build_col='Build_ID'
)
logger.info(f"✅ Prioritized test cases saved to: {prioritized_path}")

# Calculate APFD per build
apfd_path = os.path.join(results_dir, 'apfd_per_build.csv')
apfd_results_df, apfd_summary = generate_apfd_report(
    test_df_with_ranks,
    method_name=config['experiment']['name'],
    test_scenario="v8_full_test",
    output_path=apfd_path
)

# Print summary
print_apfd_summary(apfd_summary)

# Log results
if apfd_summary:
    logger.info(f"\n✅ APFD per-build report saved to: {apfd_path}")
    logger.info(f"📊 Mean APFD: {apfd_summary['mean_apfd']:.4f} (across {apfd_summary['total_builds']} builds)")

    # Verify expected 277 builds
    if apfd_summary['total_builds'] != 277:
        logger.warning(f"⚠️  WARNING: Expected 277 builds but got {apfd_summary['total_builds']}")
        logger.warning(f"   This may indicate incorrect filtering or data issues")
else:
    logger.warning("⚠️  No builds with failures found - APFD cannot be calculated")

logger.info("\n"+"="*70)
logger.info("TRAINING COMPLETE!")
logger.info("="*70)
logger.info(f"Best Val F1: {best_val_f1:.4f}")
logger.info(f"Test F1: {test_metrics['f1_macro']:.4f}")
logger.info(f"Mean APFD: {apfd_summary.get('mean_apfd', 0.0):.4f}")
```

---

## 📝 IMPORTS NECESSÁRIOS

Adicionar no início do main_v8.py:

```python
from evaluation.apfd import (
    generate_apfd_report,
    print_apfd_summary,
    generate_prioritized_csv  # ✅ ADICIONAR
)
```

---

## ✅ VALIDAÇÃO PÓS-CORREÇÃO

Após aplicar as correções, verificar:

1. **Arquivos Criados**:
   ```
   results/experiment_v8_fixed/
   ├── apfd_per_build.csv          # ✅ 277 linhas (1 por build com falha)
   └── prioritized_test_cases.csv  # ✅ Todos os TCs com ranks
   ```

2. **Colunas em prioritized_test_cases.csv**:
   - Build_ID
   - TC_Key (se disponível)
   - TE_Test_Result
   - label_binary
   - probability
   - diversity_score
   - priority_score
   - rank

3. **APFD Summary**:
   ```
   Total builds analyzed: 277
   Mean APFD: 0.XXXX
   ```

4. **Logs Esperados**:
   ```
   ✅ TE_Test_Result column found with X unique values
   ✅ Build_ID column found: YYY unique builds
   ✅ Prioritized test cases saved to: ...
   ✅ APFD per-build report saved to: ...
   📊 Mean APFD: 0.XXXX (across 277 builds)
   ```

---

## 🎯 IMPACTO DAS CORREÇÕES

### Antes (❌)
- APFD calculado com labels incorretos (incluindo Delete/Blocked)
- Arquivos salvos na raiz do projeto
- Sem CSV prioritizado
- Difícil debugging

### Depois (✅)
- APFD correto usando apenas `TE_Test_Result == 'Fail'`
- Arquivos organizados em `results/experiment_name/`
- CSV prioritizado disponível para análise
- Logs detalhados para debugging
- Validação de colunas necessárias

---

## 🚨 ATENÇÃO

**CRÍTICO**: Essas correções devem ser aplicadas **ANTES** de rodar o treino completo do modelo!

O APFD incorreto pode levar a:
- Métricas erradas reportadas em papers
- Comparações inválidas com outros métodos
- Conclusões incorretas sobre a eficácia do modelo

---

**Status**: 🔴 **CORREÇÕES NECESSÁRIAS - ALTA PRIORIDADE**

**Próxima ação**: Aplicar correções no main_v8.py antes de executar treino completo
