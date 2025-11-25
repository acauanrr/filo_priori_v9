# Migração V8 → V9: Guia Completo

## Sumário Executivo

**Problema Identificado:**
- Experimento v8_improved com fine-tuning do BGE degradou performance
- APFD caiu de 0.5967 (exp_017) para 0.5481 (v8_improved) = **-8.1%**
- Fine-tuning pode ter causado overfitting ou degradado representações gerais

**Solução Implementada:**
- Substituição do modelo de embeddings: BGE → **Qodo-Embed-1-1.5B**
- Encoding separado: TCs (summary+steps) e Commits processados independentemente
- Remoção do fine-tuning: uso do modelo pré-treinado diretamente
- Dimensão dos embeddings: 1024 → **3072** (1536 para TCs + 1536 para Commits)

---

## Análise do Experimento v8_improved

### Resultados Comparativos

| Métrica | Exp 017 (baseline) | v8_improved (fine-tuned) | Diferença |
|---------|-------------------|-------------------------|-----------|
| Mean APFD | 0.5967 | 0.5481 | **-8.1%** ❌ |
| Test F1 Macro | - | 0.5360 | - |
| Test Accuracy | - | 95.66% | - |
| Test AUPRC | - | 0.5139 | - |

### Diagnóstico
- O fine-tuning do BGE **não melhorou** a performance
- Possíveis causas:
  - Overfitting no dataset específico
  - Degradação das representações semânticas genéricas
  - Perda de capacidade de generalização

---

## Mudanças Implementadas no V9

### 1. Novo Modelo de Embeddings: Qodo-Embed-1-1.5B

**Especificações:**
- **Modelo:** `Qodo/Qodo-Embed-1-1.5B`
- **Parâmetros:** 1.5B (vs 335M do BGE)
- **Dimensão por embedding:** 1536 (vs 1024 do BGE)
- **Max tokens:** 32k (vs 512 do BGE)
- **Base:** Alibaba-NLP/gte-Qwen2-1.5B-instruct

**Vantagens:**
- Modelo maior e mais capaz (1.5B parâmetros)
- Embeddings de maior dimensão (1536 vs 1024)
- Suporte a contextos maiores (32k tokens)
- Estado-da-arte em tarefas de embedding de código
- Sem necessidade de fine-tuning

### 2. Encoding Separado de TCs e Commits

**Arquitetura Anterior (V8):**
```
[Summary + Steps + Commits] → BGE Encoder → [1024] embedding
```

**Arquitetura Nova (V9):**
```
[Summary + Steps] → Qodo Encoder → [1536] TC embedding
[Commits]         → Qodo Encoder → [1536] Commit embedding
                                   ↓
                    Concatenação → [3072] combined embedding
```

**Motivação:**
- Preserva a estrutura semântica de cada tipo de informação
- TCs e Commits têm naturezas diferentes (testes vs mudanças de código)
- Encoding separado permite ao modelo aprender representações específicas
- Dobra a capacidade representacional: 1024 → 3072

### 3. Extração e Preprocessamento de Commits

**Novo Módulo:** `src/preprocessing/commit_extractor.py`

**Funcionalidades:**
- Parse de commits em formato JSON ou string
- Limpeza de mensagens de commit
- Extração de metadados (autor, arquivos alterados)
- Formatação estruturada: `"Commit 1: <msg> | Commit 2: <msg> | ..."`
- Limite configurável de commits por TC

**Exemplo de Uso:**
```python
from preprocessing.commit_extractor import CommitExtractor

extractor = CommitExtractor(config)
commit_texts = extractor.extract_from_dataframe(df, 'commit')
```

### 4. Novo Encoder: QodoEncoder

**Novo Módulo:** `src/embeddings/qodo_encoder.py`

**Funcionalidades:**
- Carrega Qodo-Embed-1-1.5B via sentence-transformers
- Encoding separado de TCs e Commits
- Cache de embeddings por tipo (tc_embeddings.npy, commit_embeddings.npy)
- Normalização automática (recomendado para Qodo)
- API compatível com encoder anterior

**Métodos Principais:**
```python
# Encoding separado
tc_emb, commit_emb = encoder.encode_separate_embeddings(
    summaries, steps, commit_texts
)

# Encoding com cache
tc_emb, commit_emb = encoder.encode_dataset_separate(
    summaries, steps, commit_texts,
    cache_dir='cache/embeddings_qodo',
    split_name='train'
)
```

---

## Arquivos Criados/Modificados

### Novos Arquivos

1. **`src/preprocessing/commit_extractor.py`**
   - Extração e preprocessamento de commits
   - ~220 linhas

2. **`src/embeddings/qodo_encoder.py`**
   - Encoder para Qodo-Embed-1-1.5B
   - Suporte a encoding separado
   - ~260 linhas

3. **`configs/experiment_v9_qodo.yaml`**
   - Configuração completa para V9
   - Dimensões atualizadas (3072)
   - Parâmetros do Qodo-Embed

4. **`main_v9.py`**
   - Pipeline V9 com encoding separado
   - Integração do CommitExtractor e QodoEncoder
   - ~330 linhas

5. **`requirements_v9.txt`**
   - Dependências adicionais (sentence-transformers)

6. **`MIGRATION_V8_TO_V9.md`** (este arquivo)
   - Documentação completa da migração

### Arquivos Existentes (Não Modificados)

Os seguintes módulos **continuam funcionando** sem modificações:
- `src/preprocessing/data_loader.py`
- `src/preprocessing/structural_feature_extractor.py`
- `src/preprocessing/structural_feature_imputation.py`
- `src/phylogenetic/phylogenetic_graph_builder.py`
- `src/models/dual_stream_v8.py` (precisa adaptar `input_dim: 3072`)
- `src/training/` (todos os módulos)
- `src/evaluation/` (todos os módulos)

---

## Instalação e Configuração

### 1. Instalar Dependências

```bash
# Instalar sentence-transformers
pip install -r requirements_v9.txt

# Ou manualmente
pip install sentence-transformers>=2.2.2
```

### 2. Verificar Instalação

```bash
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
```

### 3. Download Automático do Modelo

Na primeira execução, o modelo Qodo-Embed-1-1.5B será baixado automaticamente:
- Tamanho: ~6GB
- Local: `~/.cache/huggingface/hub/`

---

## Como Usar

### Execução do Pipeline V9

```bash
# Execução completa
python main_v9.py --config configs/experiment_v9_qodo.yaml --device cuda

# Teste com amostra
python main_v9.py --config configs/experiment_v9_qodo.yaml --device cuda --sample 1000
```

### Estrutura de Cache

```
cache/
└── embeddings_qodo/
    ├── train_tc_embeddings.npy       # [N, 1536]
    ├── train_commit_embeddings.npy   # [N, 1536]
    ├── val_tc_embeddings.npy         # [M, 1536]
    ├── val_commit_embeddings.npy     # [M, 1536]
    ├── test_tc_embeddings.npy        # [K, 1536]
    └── test_commit_embeddings.npy    # [K, 1536]
```

---

## Adaptação do Modelo

### Configuração do Semantic Stream

O modelo `dual_stream_v8` precisa ser adaptado para aceitar `input_dim: 3072`:

```yaml
# configs/experiment_v9_qodo.yaml
model:
  semantic:
    input_dim: 3072  # 1536 (TC) + 1536 (Commit)
    hidden_dim: 256
    num_layers: 2
    dropout: 0.15
```

O `SemanticStream` já suporta qualquer `input_dim`, então a mudança é **automática** via config.

---

## Resultados Esperados

### Melhorias Esperadas

1. **Mean APFD:** 0.5481 → **0.60+** (melhor que V8)
2. **Test F1 Macro:** 0.5360 → **0.55+**
3. **Capacidade:** 1024 → **3072** (3x mais informação)
4. **Estabilidade:** Sem riscos de overfitting do fine-tuning

### Critérios de Sucesso

- ✅ Mean APFD > 0.58 (melhor que exp_017 e v8_improved)
- ✅ Test F1 Macro > 0.54
- ✅ Recall Fail ≥ 20%
- ✅ Sem colapso catastrófico
- ✅ Separação clara entre informação de TC e Commit

---

## Próximos Passos

### Integração Completa

O `main_v9.py` atual implementa apenas a **preparação de dados**. Para completar:

1. **Copiar loop de treino** do `main_v8.py`:
   - Seções de treino (STEP 3)
   - Avaliação (STEP 4)
   - Cálculo de APFD (STEP 5)

2. **Adaptar DataLoaders** para embeddings concatenados

3. **Testar pipeline completo**:
```bash
python main_v9.py --config configs/experiment_v9_qodo.yaml --device cuda
```

### Experimentos Futuros

1. **V9.1:** Testar diferentes estratégias de fusão TC+Commit
   - Concatenação (atual)
   - Attention-based fusion
   - Gated fusion

2. **V9.2:** Explorar metadados de commits
   - Incluir autor, timestamp
   - Incluir arquivos modificados
   - Análise de co-alterações

3. **V9.3:** Fine-tuning **seletivo** do Qodo
   - Apenas commit encoder
   - Apenas TC encoder
   - Comparar com pré-treinado

---

## Depreciações

### Código Removido/Depreciado

1. **Fine-tuning BGE:**
   - `configs/finetune_bge.yaml` → **OBSOLETO**
   - `configs/finetune_bge_cpu.yaml` → **OBSOLETO**
   - `src/embeddings/triplet_generator.py` → **MANTIDO** (para referência)

2. **Modelo BGE Fine-tuned:**
   - `models/finetuned_bge_v1/` → **NÃO MAIS USADO**

### Compatibilidade Retroativa

- **V8 continua funcional** com seus próprios configs
- V9 **não quebra** pipelines existentes
- Ambas versões podem coexistir

---

## Checklist de Migração

- [x] Analisar resultados v8_improved
- [x] Investigar modelo Qodo-Embed-1-1.5B
- [x] Implementar CommitExtractor
- [x] Implementar QodoEncoder
- [x] Criar configuração V9
- [x] Criar main_v9.py com encoding separado
- [x] Documentar mudanças
- [ ] Instalar sentence-transformers
- [ ] Testar encoding separado
- [ ] Integrar loop de treino completo
- [ ] Executar experimento V9 completo
- [ ] Comparar resultados V8 vs V9

---

## Contato e Suporte

Para dúvidas ou problemas:
1. Verifique logs em `logs/experiment_v9_qodo.log`
2. Consulte esta documentação
3. Revise código de `main_v9.py` e `qodo_encoder.py`

---

**Data de Criação:** 2025-11-10
**Versão:** 1.0
**Autor:** Filo-Priori V9 Team
