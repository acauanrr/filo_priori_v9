# ⚡ Comandos Rápidos: Fine-Tuning BGE

**OBJETIVO:** Criar embeddings específicos do domínio SE para melhorar o V8

---

## 🎯 Resposta Direta às Suas Perguntas

### 1. Qual script executar?

**Opção A - Script Automático (RECOMENDADO):**
```bash
bash run_finetuning.sh quick   # Teste rápido (30 min)
bash run_finetuning.sh full    # Dataset completo (10-15 horas)
```

**Opção B - Script Python Direto:**
```bash
./venv/bin/python scripts/finetune_bge.py --config configs/finetune_bge.yaml
```

### 2. Onde o modelo é salvo?

**Resposta:** `models/finetuned_bge_v1/`

✅ **Já está configurado automaticamente** - você não precisa fazer nada!

### 3. Como usar nos experimentos?

**Resposta:** Mude apenas 1 linha no config:

```yaml
# Em configs/experiment_v8_baseline.yaml (linha ~47)
semantic:
  model_name: "models/finetuned_bge_v1"  # ← Mudar esta linha
```

**Pronto!** O V8 vai usar automaticamente o modelo fine-tuned.

---

## 📋 Cenário 1: Teste Rápido (30 minutos)

**Quando usar:** Primeira vez, para testar se funciona

```bash
# 1. Executar teste rápido
bash run_finetuning.sh quick

# 2. Aguardar (~30 minutos)
# Vai aparecer progresso na tela

# 3. Quando terminar, verificar modelo
ls -lh models/finetuned_bge_v1/
```

**Resultado esperado:**
```
✓ FINE-TUNING RÁPIDO COMPLETO!
Modelo salvo em: models/finetuned_bge_v1/
```

---

## 🚀 Cenário 2: Fine-Tuning Completo (10-15 horas)

**Quando usar:** Para obter o melhor resultado

### Método Automático (RECOMENDADO):

```bash
# 1. Executar em background
bash run_finetuning.sh full

# Vai pedir confirmação:
# Continuar? (y/n): y

# 2. Anotar o PID que aparece
# Exemplo: PID: 12345

# 3. Monitorar progresso (OPCIONAL)
tail -f logs/finetune_full.log

# 4. Verificar GPU (em outro terminal)
watch -n 1 nvidia-smi

# 5. Aguardar conclusão (~10-15 horas)
```

### Método Manual (se preferir):

```bash
# 1. Editar config (mudar sample_size para null)
nano configs/finetune_bge.yaml
# Linha 27: sample_size: null

# 2. Executar em background
nohup ./venv/bin/python scripts/finetune_bge.py --config configs/finetune_bge.yaml > logs/finetune_full.log 2>&1 &

# 3. Anotar PID
echo $!

# 4. Monitorar
tail -f logs/finetune_full.log
```

---

## ✅ Cenário 3: Usar Modelo Fine-Tuned no V8

**Depois que o fine-tuning terminar:**

### Passo 1: Verificar que o modelo existe

```bash
# Deve mostrar ~1.3 GB
du -sh models/finetuned_bge_v1/

# Deve listar arquivos
ls -l models/finetuned_bge_v1/
```

### Passo 2: Atualizar config do V8

```bash
# Abrir config
nano configs/experiment_v8_baseline.yaml

# Encontrar esta linha (aproximadamente linha 47):
# model_name: "BAAI/bge-large-en-v1.5"

# Mudar para:
# model_name: "models/finetuned_bge_v1"

# Salvar (Ctrl+X, Y, Enter)
```

### Passo 3: Executar V8

```bash
# Executar com modelo fine-tuned
python main_v8.py --config configs/experiment_v8_baseline.yaml --device cuda

# O script vai automaticamente:
# 1. Detectar que model_name é um caminho local
# 2. Carregar seu modelo fine-tuned
# 3. Usar embeddings melhores
# 4. Treinar V8 com performance melhorada
```

---

## 📊 Monitoramento Durante Fine-Tuning

### Ver progresso em tempo real:

```bash
tail -f logs/finetune_full.log
```

### Ver apenas epochs:

```bash
tail -f logs/finetune_full.log | grep "Epoch"
```

### Ver GPU usage:

```bash
watch -n 1 nvidia-smi
```

### Verificar se está rodando:

```bash
ps aux | grep finetune_bge
```

---

## ⏱️ Tempo Estimado

| Configuração | Comando | Tempo | Amostras |
|--------------|---------|-------|----------|
| **Teste** | `bash run_finetuning.sh quick` | ~30 min | 10K |
| **Completo** | `bash run_finetuning.sh full` | ~10-15 horas | ~1M |

---

## 🎯 Fluxo Completo Recomendado

### 1️⃣ Primeira Execução (Teste)

```bash
# Executar teste rápido
bash run_finetuning.sh quick

# Aguardar 30 minutos

# Verificar que funcionou
ls -lh models/finetuned_bge_v1/
```

### 2️⃣ Se Teste OK → Executar Completo

```bash
# Executar fine-tuning completo
bash run_finetuning.sh full

# Confirmar: y

# Aguardar 10-15 horas (pode deixar rodando overnight)
```

### 3️⃣ Usar Modelo Fine-Tuned

```bash
# 1. Editar config
nano configs/experiment_v8_baseline.yaml
# Mudar: model_name: "models/finetuned_bge_v1"

# 2. Executar V8
python main_v8.py --config configs/experiment_v8_baseline.yaml --device cuda
```

---

## 🔥 RESUMO ULTRA-RÁPIDO

### Para começar AGORA:

```bash
# 1. Teste rápido (30 min)
bash run_finetuning.sh quick

# 2. Se funcionar, rodar completo (10-15h)
bash run_finetuning.sh full

# 3. Quando terminar, editar config:
nano configs/experiment_v8_baseline.yaml
# Linha ~47: model_name: "models/finetuned_bge_v1"

# 4. Rodar V8
python main_v8.py --config configs/experiment_v8_baseline.yaml --device cuda
```

**PRONTO!** É só isso.

---

## 📁 Arquivos Importantes

| Arquivo | Descrição |
|---------|-----------|
| `run_finetuning.sh` | Script automático (USE ESTE!) |
| `scripts/finetune_bge.py` | Script Python de fine-tuning |
| `configs/finetune_bge.yaml` | Configuração do fine-tuning |
| `logs/finetune_full.log` | Log do fine-tuning completo |
| `models/finetuned_bge_v1/` | Modelo fine-tuned (SALVO AQUI!) |

---

## 🐛 Problemas Comuns

### Erro: "CUDA out of memory"

**Solução:**
```bash
# Editar config e reduzir batch_size
nano configs/finetune_bge.yaml
# Linha 67: batch_size: 64  (de 96 para 64)
```

### Erro: "No triplets generated"

**Solução:**
```bash
# Editar config
nano configs/finetune_bge.yaml
# Linhas 33-36:
#   min_fail_builds: 1
#   min_pass_builds: 1
```

### Processo travou?

```bash
# Ver se está rodando
ps aux | grep finetune_bge

# Matar se necessário (substitua 12345 pelo PID)
kill -9 12345

# Executar novamente
bash run_finetuning.sh full
```

---

## 📞 Ajuda Rápida

```bash
# Ver guia completo
cat GUIA_EXECUCAO_FINETUNING.md

# Ver status do fine-tuning
tail -50 logs/finetune_full.log

# Ver GPU
nvidia-smi

# Ver modelo
ls -lh models/finetuned_bge_v1/
```

---

**Data:** 2025-11-06
**Status:** ✅ Pronto para Execução

**Comando para começar:**
```bash
bash run_finetuning.sh quick
```
