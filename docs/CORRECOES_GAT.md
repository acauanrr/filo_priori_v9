# Correções Implementadas - GAT e Cache

## Problemas Corrigidos

### 1. Incompatibilidade GAT com Mini-batches
**Problema**: O GAT tentava acessar índices de um grafo global (325 nós) em batches pequenos (32 amostras), causando erro "index out of bounds".

**Solução**: Implementado processamento full-graph do GAT, onde:
- Todas as features estruturais são processadas pelo GAT de uma vez
- Para cada batch, selecionamos apenas as embeddings estruturais correspondentes
- Mantém gradient flow correto para treinamento

**Arquivos alterados**:
- `main_v8.py`: Funções `train_epoch()` e `evaluate()`

### 2. Incompatibilidade de Cache com Samples
**Problema**: Quando usando `--sample-size`, os caches continham dados do dataset completo, causando:
- Embeddings: (50, 1024)
- Structural features: (54843, 6) ❌
- Labels: (54843,) ❌

**Solução**: Desabilitar caches automaticamente quando `sample_size` é especificado:
- Embeddings cache: Desabilitado em modo sample
- Structural features cache: Desabilitado em modo sample
- Phylogenetic graph cache: Desabilitado em modo sample

**Arquivos alterados**:
- `main_v8.py`: Função `prepare_data()`

### 3. Backward Through Graph Error
**Problema**: Tentativa de fazer backward múltiplas vezes através do mesmo grafo computacional.

**Solução**: Recomputar embeddings estruturais para cada batch dentro do loop de treinamento.

### 4. FocalLoss Device Mismatch
**Problema**: O `criterion` (FocalLoss) não era movido para o dispositivo (GPU), causando erro: "indices should be either on cpu or on the same device as the indexed tensor".

**Solução**: Adicionar `.to(device)` ao criterion durante inicialização.

**Arquivos alterados**:
- `main_v8.py`: Linha 415 - `criterion = FocalLoss(...).to(device)`

## Comandos de Execução

### Opção 1: Script Automatizado (Recomendado)
```bash
./run_experiment_v8.sh
```

Este script:
- Pergunta se quer limpar caches (recomendado)
- Detecta automaticamente o Python correto
- Executa com GPU (cuda)
- Mostra progresso e salva resultados

### Opção 2: Comando Manual - Treino Completo
```bash
# Limpar caches primeiro
rm -rf cache/embeddings/*.npy
rm -f cache/structural_features.pkl
rm -f cache/phylogenetic_graph.pkl

# Executar (detecte seu python correto)
python main_v8.py --config configs/experiment_v8_baseline.yaml --device cuda
```

### Opção 3: Teste Rápido com Sample
```bash
# Caches são automaticamente desabilitados com --sample-size
python main_v8.py --config configs/experiment_v8_baseline.yaml --device cpu --sample-size 1000
```

### Opção 4: Rodar em Background com tmux
```bash
# Limpar caches
rm -rf cache/embeddings/*.npy cache/*.pkl

# Iniciar sessão tmux
tmux new-session -d -s v8_baseline "python main_v8.py --config configs/experiment_v8_baseline.yaml --device cuda"

# Monitorar progresso
tmux attach -t v8_baseline

# Detach: Ctrl+B, depois D
```

## Configuração Atual

### Modelo Fine-tuned
- **Localização**: `models/finetuned_bge_v1`
- **Base**: BAAI/bge-large-en-v1.5
- **Fine-tuning**: Triplet loss com dados do projeto
- **Config**: `configs/experiment_v8_baseline.yaml` (linha 52)

### Arquitetura
- **Semantic Stream**: BGE embeddings (1024-dim) → FFN (256-dim)
- **Structural Stream**: 6 features históricas → GAT (4 heads) → 256-dim
- **Fusion**: Cross-attention bidirectional
- **Classifier**: [512 → 128 → 64 → 2]

### Features Estruturais (6 total)
1. `test_age`: Idade do teste (builds desde primeira aparição)
2. `failure_rate`: Taxa de falha histórica
3. `recent_failure_rate`: Taxa de falha recente (última janela)
4. `flakiness_rate`: Taxa de oscilação Pass↔Fail
5. `commit_count`: Número de commits associados
6. `test_novelty`: Flag de primeira aparição

### Grafo Filogenético
- **Tipo**: Co-failure (testes que falharam juntos)
- **Nós**: 325 test cases
- **Arestas**: 265 conexões
- **Processamento**: GATConv com 4 attention heads

## Resultados Esperados

### Métricas Alvo
- **Test F1 Macro**: ≥ 0.60
- **Test Accuracy**: ≥ 0.70
- **APFD Mean**: ≥ 0.75

### Outputs Salvos
- `results/experiment_v8_baseline/best_model.pt`: Melhor modelo
- `results/experiment_v8_baseline/test_metrics.json`: Métricas finais
- `results/experiment_v8_baseline/confusion_matrix.png`: Matriz de confusão
- `results/experiment_v8_baseline/precision_recall_curves.png`: Curvas PR
- `results/experiment_v8_baseline/prioritized_test_cases.csv`: Testes priorizados
- `results/experiment_v8_baseline/apfd_per_build.csv`: APFD por build

## Troubleshooting

### Erro: "Size mismatch between tensors"
**Causa**: Caches desatualizados com shapes incompatíveis
**Solução**:
```bash
rm -rf cache/embeddings/*.npy cache/*.pkl
```

### Erro: "index out of bounds for dimension 0"
**Causa**: Problema de batching com GAT (já corrigido)
**Solução**: Use a versão atualizada do `main_v8.py`

### Erro: "indices should be either on cpu or on the same device"
**Causa**: FocalLoss não foi movido para o device correto (já corrigido)
**Solução**: Use a versão atualizada do `main_v8.py`

### Erro: "No module named 'torch'"
**Causa**: Ambiente virtual não encontrado ou não ativado
**Solução**:
```bash
# Criar venv se não existir
python3 -m venv venv

# Instalar dependências
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install sentence-transformers

# Usar venv
./venv/bin/python main_v8.py ...
```

### Warning: "NVML_SUCCESS == DriverAPI"
**Causa**: Problema com drivers CUDA/GPU
**Solução**: Use CPU em vez de CUDA
```bash
python main_v8.py --config configs/experiment_v8_baseline.yaml --device cpu
```

## Próximos Passos

1. **Executar Experimento**: `./run_experiment_v8.sh`
2. **Analisar Resultados**: Verificar métricas em `results/experiment_v8_baseline/`
3. **Comparar com Baseline**: Comparar com resultados V7
4. **Ajustar Hiperparâmetros**: Se necessário, ajustar learning rate, dropout, etc.
5. **Experimentos Adicionais**: Testar outras configurações (gated fusion, etc.)

## Status

✅ GAT mini-batch incompatibility - **CORRIGIDO**
✅ Cache size mismatch - **CORRIGIDO**
✅ Backward through graph error - **CORRIGIDO**
✅ FocalLoss device mismatch - **CORRIGIDO**
✅ Modelo fine-tuned configurado - **OK**
✅ Script de execução criado - **OK**

**Pronto para execução!** 🚀
