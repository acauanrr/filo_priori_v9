# Configurações - configs/

**Última atualização:** 2025-11-05
**Status:** ✅ LIMPO E ORGANIZADO

---

## 📂 Estrutura Atual

Este diretório contém **12 configurações ativas** usadas pelos experimentos do projeto.

### Experimentos Principais (9 configs):

| Arquivo | Experimento | Status | Resultados |
|---------|-------------|--------|------------|
| `experiment_008_gatv2.yaml` | GATv2 Implementation | ✅ Executado | `results/experiment_008_gatv2/` |
| `experiment_009_attention_pooling.yaml` | Attention Pooling | ✅ Executado | `results/experiment_009_attention_pooling/` |
| `experiment_010_bidirectional_fusion.yaml` | Bidirectional Fusion | ✅ Executado | `results/experiment_010_bidirectional_fusion/` |
| `experiment_011_improved_classifier.yaml` | Improved Classifier | ✅ Executado | `results/experiment_011_improved_classifier/` |
| `experiment_012_best_practices.yaml` | Best Practices | ✅ Executado | `results/experiment_012_best_practices/` |
| `experiment_014_ranking_fix.yaml` | Ranking Fix | ✅ Executado | `results/experiment_014_ranking_fix/` |
| `experiment_015_gatv2_rewired.yaml` | GATv2 + Rewired Graph | ✅ Executado | `results/experiment_015_gatv2_rewired/` |
| `experiment_016_optimized.yaml` | Optimized Loss & Rewiring | ✅ Executado | `results/experiment_016_optimized_*` |
| `experiment_017_ranking_corrected.yaml` | Ranking Corrected | ✅ **ATUAL** | `results/experiment_017_ranking_corrected_*` |

### Rewiring Configs (3 configs):

| Arquivo | Usado Por | Descrição |
|---------|-----------|-----------|
| `rewiring_015.yaml` | Exp 015 | k=10, keep_ratio=0.0 |
| `rewiring_016.yaml` | Exp 016 | k=15, keep_ratio=0.2 |
| `rewiring_017.yaml` | Exp 017 | k=20, keep_ratio=0.2 |

---

## 🎯 Config Atual

**Experimento mais recente:** `experiment_017_ranking_corrected.yaml`

**Usado em:**
- `run_experiment_017.sh` (script principal)
- `main.py` (default atualizado)

**Características:**
- Focal loss: [0.15, 0.85]
- Rewiring: k=20, keep_ratio=0.2
- Binary classification: Pass (0) vs Fail (1)
- Multi-field embeddings
- GATv2 + Rewired graph

---

## 📖 Uso

### Executar Experimento Atual:

```bash
# Usando default (exp 017)
python main.py

# Ou explicitamente
python main.py --config configs/experiment_017_ranking_corrected.yaml
```

### Executar Experimento Específico:

```bash
# Experimento 015
python main.py --config configs/experiment_015_gatv2_rewired.yaml

# Experimento 012
python main.py --config configs/experiment_012_best_practices.yaml
```

### Executar com Script de Shell:

```bash
# Experimento 017 (completo com rewiring)
./run_experiment_017.sh
```

---

## 🗂️ Arquivos Arquivados

### Configs Obsoletos:

Movidos para `archive_old/configs/obsolete/` (12 arquivos):
- `config.yaml`
- `config_experiment_003.yaml`
- `config_experiment_004.yaml`
- `config_experiment_004_moderate.yaml`
- `config_experiment_006.yaml`
- `config_improved.yaml`
- `experiment_009_denoising_gate.yaml`
- `experiment_009b_adaptive_denoising.yaml`
- `experiment_010_graph_rewiring.yaml`
- `experiment_013_pass_vs_fail.yaml`
- `experiment_017_ranking_margin.yaml`
- `rewiring_config.yaml`

**Motivo:** Sem resultados correspondentes ou substituídos por versões mais recentes.

### Configs de Planejamento:

Movidos para `docs/phases/` (3 arquivos):
- `phase_1_stabilization.yaml`
- `phase_2_architectural_refinement.yaml`
- `phase_3_hyperparameter_optimization.yaml`

**Motivo:** Documentação de planejamento, não configs executáveis.

---

## 📋 Estrutura de um Config

Exemplo de estrutura típica:

```yaml
# Dados
data:
  train_path: "datasets/train.csv"
  test_path: "datasets/test.csv"
  output_dir: "results/experiment_XXX/"
  binary_classification: true

# Embeddings
embedding:
  model_name: "BAAI/bge-large-en-v1.5"
  use_multi_field: true
  fields: [summary, steps, commits, CR]

# Modelo
model:
  semantic_stream: {...}
  structural_stream:
    layer_type: "gatv2"
    num_gnn_layers: 2

# Treinamento
training:
  num_epochs: 20
  batch_size: 64
  learning_rate: 5.0e-5
  loss:
    type: "focal"
    focal_alpha: [0.15, 0.85]

# Hardware
hardware:
  device: "cuda"
```

---

## 🔄 Histórico de Limpeza

**2025-11-05:**
- ✅ Limpeza inicial: 27 → 12 configs (redução de 55%)
- ✅ Arquivados 12 configs obsoletos
- ✅ Movidos 3 configs de planejamento para docs/
- ✅ Atualizado default em `main.py`
- ✅ Criado `README.md` (este arquivo)

---

## 📚 Documentação Relacionada

- **Análise Completa:** `CONFIG_CLEANUP_ANALYSIS.md` (raiz do projeto)
- **Refatoração Geral:** `REFACTORING_SUMMARY.md`
- **Status Implementação:** `IMPLEMENTATION_COMPLETE.md`
- **Planejamento de Fases:** `docs/phases/phase_*.yaml`

---

## ⚠️ Notas Importantes

### 1. Criar Novo Experimento:

Ao criar novo experimento (ex: 018):
1. Copiar config mais recente como base
2. Modificar parâmetros necessários
3. Atualizar `experiment.experiment_id` no YAML
4. Salvar como `experiment_018_<descricao>.yaml`

### 2. Rewiring Configs:

Configs de rewiring são sempre separados e referenciados pelos experimentos principais.

### 3. Manter Limpo:

- ✅ Não criar configs temporários
- ✅ Arquivar configs de experimentos abandonados
- ✅ Manter nomenclatura consistente: `experiment_XXX_<descricao>.yaml`

---

## 🆘 Suporte

**Dúvidas sobre configs?**
- Ver documentação completa em `CONFIG_CLEANUP_ANALYSIS.md`
- Consultar experimentos anteriores em `results/`
- Verificar logs de execução

**Restaurar config arquivado:**
```bash
# Copiar de archive_old para configs/
cp archive_old/configs/obsolete/config_XXX.yaml configs/
```

---

**Mantido por:** Equipe do Projeto
**Última verificação:** 2025-11-05
