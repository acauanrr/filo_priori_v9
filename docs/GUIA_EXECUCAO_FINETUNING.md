# 🚀 Guia de Execução: Fine-Tuning do BGE para Domínio SE

**Data:** 2025-11-06
**Objetivo:** Fine-tuning do BGE-Large para criar embeddings específicos do domínio de Software Engineering

---

## 📋 Visão Geral

### O que vai acontecer?

1. **Geração de Triplets**: O script vai analisar o histórico de execução de testes e gerar triplets:
   - **Anchor**: Texto do caso de teste (TE_Summary + TC_Steps)
   - **Positive**: Commit que causou falha no teste
   - **Negative**: Commit em builds onde o teste passou

2. **Fine-Tuning**: Treinar o BGE-Large usando TripletLoss para aprender relações SE-específicas

3. **Salvamento Automático**: Modelo fine-tuned será salvo em `models/finetuned_bge_v1/`

4. **Integração**: Você só precisa mudar **1 linha** no config para usar o modelo fine-tuned

### Tempo Estimado

| Configuração | Tempo | Triplets | Uso |
|--------------|-------|----------|-----|
| **Teste Rápido** | ~30 min | 10K | Validar pipeline |
| **Médio** | ~3-4 horas | 100K | Bom resultado |
| **Completo** | ~10-15 horas | 1M+ | Melhor resultado |

---

## 🔧 PASSO 1: Instalar Dependências

Execute o script de setup (já está pronto):

```bash
# Ativar ambiente virtual
source venv/bin/activate

# Executar setup (instala sentence-transformers)
bash setup_finetuning.sh
```

**O que ele instala:**
- `sentence-transformers`: Biblioteca para fine-tuning de embeddings
- Dependências adicionais necessárias

**Verificação:**
```bash
# Verificar se instalou corretamente
./venv/bin/python -c "import sentence_transformers; print('✓ OK')"
```

---

## 🧪 PASSO 2: Testar Geração de Triplets (OPCIONAL mas Recomendado)

Antes de rodar o fine-tuning completo, teste se a geração de triplets funciona:

```bash
# Testar geração de triplets
./venv/bin/python scripts/test_triplet_generation.py
```

**O que esperar:**
```
✓ Carregando dados de treino...
✓ Gerando triplets...
✓ Total de triplets gerados: 15,432
✓ Exemplos de triplets:

Anchor:   "TE - TC - Login: User login with valid credentials"
Positive: "Fix authentication bug in login module"
Negative: "Update README documentation"

✓ Cache salvo em: cache/triplets_test.csv
```

**Se der erro aqui:** Significa que há problema com os dados. Resolva antes de continuar.

---

## ⚡ PASSO 3: Fine-Tuning RÁPIDO (Teste de 30 minutos)

**RECOMENDADO PRIMEIRO!** Teste o pipeline completo com amostra pequena:

### 3.1. Editar configuração para teste rápido

Abra o arquivo de configuração:
```bash
nano configs/finetune_bge.yaml
```

Mude a linha 27 de:
```yaml
sample_size: null  # Use full dataset
```

Para:
```yaml
sample_size: 10000  # Para teste rápido
```

### 3.2. Executar fine-tuning de teste

```bash
# Executar fine-tuning (30 minutos)
./venv/bin/python scripts/finetune_bge.py --config configs/finetune_bge.yaml
```

### 3.3. Monitorar progresso

Em outro terminal, monitore a GPU:
```bash
watch -n 1 nvidia-smi
```

**O que você vai ver:**
```
======================================================================
FINE-TUNING BGE FOR SOFTWARE ENGINEERING DOMAIN
======================================================================

Loading base model: BAAI/bge-large-en-v1.5...
✓ Model loaded

Generating triplets from training data...
✓ Generated 8,543 triplets from 10,000 samples
✓ Cache saved to: cache/triplets_full.csv

Training configuration:
  - Epochs: 5
  - Batch size: 96
  - Learning rate: 3e-5
  - Device: cuda (Quadro RTX 8000)

Training...
Epoch 1/5: 100%|██████████| 89/89 [02:15<00:00, 0.66it/s] Loss: 0.4521
Epoch 2/5: 100%|██████████| 89/89 [02:14<00:00, 0.66it/s] Loss: 0.3124
Epoch 3/5: 100%|██████████| 89/89 [02:15<00:00, 0.66it/s] Loss: 0.2456
Epoch 4/5: 100%|██████████| 89/89 [02:14<00:00, 0.66it/s] Loss: 0.1987
Epoch 5/5: 100%|██████████| 89/89 [02:15<00:00, 0.66it/s] Loss: 0.1654

✓ Training complete!

Saving model to: models/finetuned_bge_v1/
✓ Model saved successfully!

======================================================================
FINE-TUNING COMPLETE
======================================================================
Time elapsed: 11m 15s
```

### 3.4. Verificar modelo salvo

```bash
# Verificar se o modelo foi salvo
ls -lh models/finetuned_bge_v1/

# Você deve ver:
# config.json
# pytorch_model.bin
# tokenizer_config.json
# ...
```

**Se funcionou:** Parabéns! Agora pode rodar o fine-tuning completo.

**Se deu erro:** Verifique os logs em `logs/finetune_bge.log`

---

## 🏆 PASSO 4: Fine-Tuning COMPLETO (10-15 horas)

Agora execute o fine-tuning com o **dataset completo** para obter o melhor resultado.

### 4.1. Editar configuração para dataset completo

Abra o arquivo de configuração:
```bash
nano configs/finetune_bge.yaml
```

Mude a linha 27 de:
```yaml
sample_size: 10000  # Para teste rápido
```

Para:
```yaml
sample_size: null  # Use full dataset
```

Salve e feche (Ctrl+X, Y, Enter).

### 4.2. Criar diretório de logs (se não existir)

```bash
mkdir -p logs
```

### 4.3. Executar fine-tuning em background

**IMPORTANTE:** Como vai demorar 10-15 horas, execute em background:

```bash
# Executar em background com nohup
nohup ./venv/bin/python scripts/finetune_bge.py --config configs/finetune_bge.yaml > logs/finetune_full.log 2>&1 &

# Anotar o PID (process ID) que aparece
# Exemplo: [1] 12345
```

### 4.4. Monitorar progresso

**Opção 1: Ver log em tempo real**
```bash
tail -f logs/finetune_full.log
```

**Opção 2: Verificar últimas 50 linhas**
```bash
tail -50 logs/finetune_full.log
```

**Opção 3: Monitorar GPU**
```bash
watch -n 1 nvidia-smi
```

### 4.5. Verificar se ainda está rodando

```bash
# Ver processos python
ps aux | grep finetune_bge

# Verificar GPU usage
nvidia-smi
```

### 4.6. Aguardar conclusão

**Sinais de que terminou:**
1. GPU usage volta para 0%
2. Log mostra "FINE-TUNING COMPLETE"
3. Processo não aparece mais em `ps aux`

---

## 📁 Onde o Modelo é Salvo?

### Localização Padrão (Configurada Automaticamente)

```
models/finetuned_bge_v1/
├── config.json                  # Configuração do modelo
├── pytorch_model.bin            # Pesos do modelo fine-tuned
├── tokenizer_config.json        # Configuração do tokenizer
├── vocab.txt                    # Vocabulário
├── special_tokens_map.json      # Tokens especiais
└── 1_Pooling/
    └── config.json              # Configuração de pooling
```

**Tamanho esperado:** ~1.3 GB

### Verificar Modelo Salvo

```bash
# Ver tamanho do modelo
du -sh models/finetuned_bge_v1/

# Listar arquivos
ls -lh models/finetuned_bge_v1/

# Testar se o modelo carrega
./venv/bin/python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('models/finetuned_bge_v1')
print('✓ Modelo fine-tuned carrega corretamente!')
"
```

---

## 🔗 Como Usar o Modelo Fine-Tuned nos Experimentos V8

### Integração Automática (Muito Simples!)

O modelo fine-tuned já está salvo no local correto. Você só precisa **mudar 1 linha** no config:

#### ANTES (usando BGE genérico):
```yaml
# configs/experiment_v8_baseline.yaml
semantic:
  model_name: "BAAI/bge-large-en-v1.5"  # BGE genérico
  embedding_dim: 1024
  ...
```

#### DEPOIS (usando BGE fine-tuned):
```yaml
# configs/experiment_v8_baseline.yaml ou experiment_v8_gated_fusion.yaml
semantic:
  model_name: "models/finetuned_bge_v1"  # Seu modelo fine-tuned!
  embedding_dim: 1024
  ...
```

### Executar Experimento V8 com Modelo Fine-Tuned

```bash
# Executar V8 com modelo fine-tuned
python main_v8.py --config configs/experiment_v8_baseline.yaml --device cuda
```

**O script vai automaticamente:**
1. Detectar que `model_name` aponta para pasta local
2. Carregar seu modelo fine-tuned de `models/finetuned_bge_v1/`
3. Usar os embeddings fine-tuned para extrair features semânticas
4. Treinar o modelo V8 com embeddings melhores

**Sem código adicional necessário!** Tudo já está integrado.

---

## 📊 Comparação de Resultados Esperados

### Baseline (BGE Genérico)
```
Test F1 Macro:  0.55-0.60
Test Accuracy:  65-70%
Mean APFD:      0.60-0.65
```

### Com Fine-Tuned (BGE Domínio SE)
```
Test F1 Macro:  0.60-0.65  (+5-10pp) ✓
Test Accuracy:  70-75%     (+5pp)     ✓
Mean APFD:      0.65-0.70  (+5-10pp) ✓
```

**Melhoria esperada:** +5 a 10 pontos percentuais em todas as métricas!

---

## 🐛 Troubleshooting (Soluções de Problemas)

### Problema 1: Out of Memory (OOM)

**Erro:**
```
RuntimeError: CUDA out of memory
```

**Solução:**
Edite `configs/finetune_bge.yaml` e reduza `batch_size`:
```yaml
training:
  batch_size: 64  # De 96 para 64
```

### Problema 2: No Triplets Generated

**Erro:**
```
Error: No triplets generated! Check your data...
```

**Causa:** Dados não têm Pass e Fail suficientes

**Solução:**
Edite `configs/finetune_bge.yaml`:
```yaml
triplet:
  min_fail_builds: 1  # Reduzir de 2 para 1
  min_pass_builds: 1  # Reduzir de 2 para 1
```

### Problema 3: Processo Travou

**Verificar se está rodando:**
```bash
ps aux | grep finetune_bge
nvidia-smi
```

**Matar processo se necessário:**
```bash
# Encontrar PID
ps aux | grep finetune_bge

# Matar (substitua 12345 pelo PID real)
kill -9 12345
```

### Problema 4: Modelo não Carrega

**Verificar integridade:**
```bash
# Verificar se arquivos existem
ls -l models/finetuned_bge_v1/pytorch_model.bin

# Verificar tamanho (deve ser ~1.3GB)
du -h models/finetuned_bge_v1/pytorch_model.bin
```

**Se corrompido:**
- Delete a pasta: `rm -rf models/finetuned_bge_v1/`
- Execute o fine-tuning novamente

---

## 📝 Resumo dos Comandos

### Setup (Primeira Vez)
```bash
# 1. Instalar dependências
bash setup_finetuning.sh

# 2. Testar geração de triplets
./venv/bin/python scripts/test_triplet_generation.py
```

### Teste Rápido (30 minutos)
```bash
# 1. Editar config: sample_size: 10000
nano configs/finetune_bge.yaml

# 2. Executar
./venv/bin/python scripts/finetune_bge.py --config configs/finetune_bge.yaml
```

### Fine-Tuning Completo (10-15 horas)
```bash
# 1. Editar config: sample_size: null
nano configs/finetune_bge.yaml

# 2. Criar logs
mkdir -p logs

# 3. Executar em background
nohup ./venv/bin/python scripts/finetune_bge.py --config configs/finetune_bge.yaml > logs/finetune_full.log 2>&1 &

# 4. Monitorar
tail -f logs/finetune_full.log
watch -n 1 nvidia-smi
```

### Usar Modelo Fine-Tuned
```bash
# 1. Editar config do V8
nano configs/experiment_v8_baseline.yaml
# Mudar: model_name: "models/finetuned_bge_v1"

# 2. Executar experimento
python main_v8.py --config configs/experiment_v8_baseline.yaml --device cuda
```

---

## ✅ Checklist

### Antes de Começar
- [ ] `setup_finetuning.sh` executado
- [ ] `sentence-transformers` instalado
- [ ] GPU disponível (nvidia-smi funcionando)
- [ ] Espaço em disco: ~10GB livre

### Teste Rápido (30 min)
- [ ] Config editado: `sample_size: 10000`
- [ ] Fine-tuning executado sem erros
- [ ] Modelo salvo em `models/finetuned_bge_v1/`
- [ ] Modelo carrega corretamente

### Fine-Tuning Completo (10-15 horas)
- [ ] Config editado: `sample_size: null`
- [ ] Processo rodando em background
- [ ] GPU sendo utilizada (~90-100%)
- [ ] Log sendo atualizado
- [ ] Aguardar conclusão (~10-15 horas)

### Integração com V8
- [ ] Modelo fine-tuned existe em `models/finetuned_bge_v1/`
- [ ] Config V8 atualizado: `model_name: "models/finetuned_bge_v1"`
- [ ] Experimento V8 executado com modelo fine-tuned
- [ ] Resultados comparados (baseline vs fine-tuned)

---

## 🎯 Próximos Passos Após Fine-Tuning

1. **Executar V8 Baseline** (sem fine-tuning)
   ```bash
   # model_name: "BAAI/bge-large-en-v1.5"
   python main_v8.py --config configs/experiment_v8_baseline.yaml --device cuda
   ```

2. **Executar V8 com Fine-Tuned**
   ```bash
   # model_name: "models/finetuned_bge_v1"
   python main_v8.py --config configs/experiment_v8_baseline.yaml --device cuda
   ```

3. **Comparar Resultados**
   - F1 Macro
   - APFD
   - Accuracy
   - Recall de Pass/Not-Pass

4. **Executar Variantes**
   - V8 + Fine-tuned + Cross-Attention
   - V8 + Fine-tuned + Gated Fusion
   - Ablation studies

---

## 📞 Contato e Suporte

**Logs importantes:**
- Fine-tuning: `logs/finetune_full.log`
- TensorBoard: `runs/finetune_bge_v1/`
- Checkpoints: `models/finetuned_bge_v1/`

**Comandos úteis:**
```bash
# Ver progresso
tail -f logs/finetune_full.log | grep "Epoch"

# Ver loss
tail -f logs/finetune_full.log | grep "Loss"

# Ver GPU
nvidia-smi

# Ver tamanho do modelo
du -sh models/finetuned_bge_v1/
```

---

**Data de Criação:** 2025-11-06
**Última Atualização:** 2025-11-06
**Status:** ✅ Pronto para Execução

**Dúvidas?** Consulte `STEP_2.3_FINETUNING_GUIDE.md` para detalhes técnicos.
