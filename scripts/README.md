# Scripts Utilitários - scripts/

**Última atualização:** 2025-11-05
**Status:** ✅ LIMPO E ORGANIZADO

---

## 📂 Estrutura Atual

Este diretório contém **7 scripts utilitários** ativos usados para análise, validação e manutenção do projeto.

### Utilitários (3 scripts):

| Script | Função | Uso |
|--------|--------|-----|
| `cleanup_project.sh` | Limpa arquivos temporários e cache | `./scripts/cleanup_project.sh` |
| `compare_experiments_quick.sh` | Compara métricas entre experimentos | `./scripts/compare_experiments_quick.sh` |
| `extract_all_metrics.py` | Extrai e consolida métricas de experimentos | `python scripts/extract_all_metrics.py` |

### Validação e Testes (4 scripts):

| Script | Função | Uso |
|--------|--------|-----|
| `validate_experiment_015.py` | Valida integridade do experimento 015 | `python scripts/validate_experiment_015.py` |
| `validate_experiment_015_static.py` | Validação estática do experimento 015 | `python scripts/validate_experiment_015_static.py` |
| `test_gatv2_implementation.py` | Testa implementação do GATv2 layer | `python scripts/test_gatv2_implementation.py` |
| `test_load_full_testcsv.py` | Testa carregamento do test.csv completo | `python scripts/test_load_full_testcsv.py` |

---

## 🧹 cleanup_project.sh

**Propósito:** Remove arquivos temporários, caches e outputs intermediários.

**Uso:**
```bash
./scripts/cleanup_project.sh
```

**O que remove:**
- `__pycache__/` e `*.pyc`
- `*.log` temporários
- Caches de embeddings
- Checkpoints temporários
- Arquivos `.DS_Store`

**Quando usar:**
- Antes de commits importantes
- Quando o projeto está ocupando muito espaço
- Para limpar após experimentos

---

## 📊 compare_experiments_quick.sh

**Propósito:** Compara métricas de múltiplos experimentos rapidamente.

**Uso:**
```bash
./scripts/compare_experiments_quick.sh
```

**Output:**
- Tabela comparativa de métricas
- Test Accuracy, F1 Macro, AUPRC, Mean APFD
- Ordenado por performance

**Exemplo de output:**
```
=== Comparison of Experiments ===
Exp 015: Acc=0.65, F1=0.55, APFD=0.68
Exp 016: Acc=0.67, F1=0.57, APFD=0.70
Exp 017: Acc=0.68, F1=0.58, APFD=0.72 ⭐ BEST
```

---

## 📈 extract_all_metrics.py

**Propósito:** Extrai métricas de todos os experimentos e gera relatório consolidado.

**Uso:**
```bash
python scripts/extract_all_metrics.py
```

**Output:**
- CSV com todas as métricas: `all_experiments_metrics.csv`
- Inclui: accuracy, precision, recall, F1, AUPRC, APFD
- Um experimento por linha

**Colunas geradas:**
- experiment_id
- test_accuracy
- test_f1_macro
- test_auprc_macro
- mean_apfd
- builds_analyzed

---

## ✅ validate_experiment_015.py

**Propósito:** Valida que o experimento 015 está completo e correto.

**Uso:**
```bash
python scripts/validate_experiment_015.py
```

**Verificações:**
- Config existe e é válido
- Modelo treinado existe
- Resultados existem
- APFD calculado
- Métricas consistentes

**Output:**
```
✅ Config encontrado
✅ Modelo existe (best_model.pt)
✅ Resultados completos
✅ APFD calculado (277 builds)
✅ Experimento 015 VÁLIDO
```

---

## 🔍 validate_experiment_015_static.py

**Propósito:** Validação estática (sem executar) do experimento 015.

**Uso:**
```bash
python scripts/validate_experiment_015_static.py
```

**Diferença do validate_experiment_015.py:**
- Não carrega modelos (mais rápido)
- Apenas verifica existência de arquivos
- Valida estrutura de diretórios

---

## 🧪 test_gatv2_implementation.py

**Propósito:** Testa a implementação do GATv2 layer.

**Uso:**
```bash
python scripts/test_gatv2_implementation.py
```

**Testes:**
- Forward pass funciona
- Backward pass funciona
- Dimensões de output corretas
- Atenção calculada corretamente

**Quando usar:**
- Após modificar GATv2 layer
- Para debugging de problemas de gradiente
- Testes de regressão

---

## 📦 test_load_full_testcsv.py

**Propósito:** Testa carregamento do dataset test.csv completo.

**Uso:**
```bash
python scripts/test_load_full_testcsv.py
```

**Verificações:**
- test.csv existe
- Carrega corretamente
- Número de builds correto (277)
- Colunas esperadas presentes

---

## 🗑️ Scripts Arquivados

Scripts obsoletos foram movidos para `archive_old/scripts/obsolete/`:

### APFD Scripts (7 arquivos):
Movidos para `archive_old/scripts/obsolete/apfd/`:
- `calculate_apfd_277_builds.py`
- `calculate_apfd_experiment_012.py`
- `calculate_apfd_on_full_test.py`
- `recalculate_apfd_exp012.py`
- `recalculate_apfd_on_test_csv.py`
- `calculate_apfd_full_test.sh`
- `run_apfd_277_builds.sh`

**Motivo:** Substituídos por `src/evaluation/apfd_calculator.py` (mais robusto e centralizado)

### Experimento 014 Scripts (3 arquivos):
Movidos para `archive_old/scripts/obsolete/experiments/`:
- `run_experiment_014.sh`
- `verify_experiment_014_setup.sh`
- `extract_metrics.sh`

**Motivo:** Experimento 014 completo, scripts não mais necessários. Experimento atual é 017.

---

## 🔄 Histórico de Limpeza

**2025-11-05:**
- ✅ Limpeza inicial: 18 → 7 scripts (redução de 61%)
- ✅ Arquivados 7 scripts APFD obsoletos
- ✅ Arquivados 3 scripts de experimento 014
- ✅ Movido log para results/experiment_014_ranking_fix/logs/
- ✅ Criado README.md (este arquivo)

**Redução de código redundante:**
- Scripts APFD antigos: ~1.473 linhas
- Novo apfd_calculator.py: 428 linhas
- **Economia:** 71% menos código, funcionalidade superior

---

## 🚀 Uso Recomendado

### Limpeza Regular:
```bash
# Limpar projeto semanalmente
./scripts/cleanup_project.sh
```

### Comparar Experimentos:
```bash
# Ver qual experimento tem melhor performance
./scripts/compare_experiments_quick.sh
```

### Extrair Métricas:
```bash
# Gerar relatório consolidado
python scripts/extract_all_metrics.py
```

### Validar Experimento:
```bash
# Antes de usar resultados de um experimento
python scripts/validate_experiment_015.py
```

### Testar Componentes:
```bash
# Após modificar GATv2
python scripts/test_gatv2_implementation.py
```

---

## 📚 Documentação Relacionada

- **Análise Completa:** `SCRIPTS_CLEANUP_ANALYSIS.md` (raiz do projeto)
- **Limpeza de Configs:** `CONFIG_CLEANUP_COMPLETE.md`
- **Refatoração Geral:** `REFACTORING_SUMMARY.md`
- **Cálculo de APFD:** `src/evaluation/apfd_calculator.py`

---

## 🔮 Plano Futuro

### Migração para tests/ (planejado):

Quando criar estrutura formal de testes:
```
tests/
├── test_apfd_calculator.py     (mover de scripts/)
├── test_gatv2.py               (mover test_gatv2_implementation.py)
├── test_data_loading.py        (mover test_load_full_testcsv.py)
└── validate_experiments.py     (consolidar validates)
```

### Scripts Utilitários Permanecem:

```
scripts/
├── cleanup_project.sh          (manter)
├── compare_experiments.sh      (manter)
├── extract_metrics.py          (manter)
└── README.md
```

---

## ⚠️ Notas Importantes

### 1. Cálculo de APFD

**NÃO use os scripts antigos em archive_old/!**

Use sempre:
```python
from src.evaluation.apfd_calculator import APFDCalculator

# Calcular APFD
results = APFDCalculator.calculate_modified_apfd(df_ordered)
print(f"APFD: {results['apfd']:.4f}")
```

Ou via main.py (integrado):
```bash
python main.py --config configs/experiment_017_ranking_corrected.yaml
# APFD calculado automaticamente
```

### 2. Executar Experimentos

**NÃO use run_experiment_014.sh!**

Use o experimento atual:
```bash
./run_experiment_017.sh
```

### 3. Validação

Scripts de validação são específicos para experimento 015. Para validar outros experimentos, adapte conforme necessário.

---

## 🆘 Suporte

**Dúvidas sobre scripts?**
- Ver análise completa em `SCRIPTS_CLEANUP_ANALYSIS.md`
- Consultar código-fonte dos scripts
- Verificar logs de execução

**Restaurar script arquivado:**
```bash
# Copiar de archive_old para scripts/
cp archive_old/scripts/obsolete/apfd/calculate_apfd_277_builds.py scripts/
```

**Problema com script?**
- Verificar se tem permissões de execução: `chmod +x scripts/script.sh`
- Ver documentação no cabeçalho do script
- Testar em ambiente isolado primeiro

---

## 📊 Estatísticas

**Scripts ativos:** 7
**Scripts arquivados:** 10
**Redução:** 61% (de 18 para 7)
**Código APFD:** Reduzido de ~1.473 para 428 linhas (71% menos)

**Benefícios:**
- ✅ Menos redundância
- ✅ Código centralizado
- ✅ Mais fácil de manter
- ✅ Melhor organização

---

**Mantido por:** Equipe do Projeto
**Última verificação:** 2025-11-05
