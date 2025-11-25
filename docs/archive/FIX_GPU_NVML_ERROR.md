# FIX: Erro de GPU - NVML Initialization Failed

## 🔴 Problema

```
RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED
at "../c10/cuda/CUDACachingAllocator.cpp":963
```

**Warning precedente**:
```
UserWarning: Can't initialize NVML
```

## 🔍 Causa

Este é um erro do ambiente WSL2/CUDA, não do código. A NVIDIA Management Library (NVML) não consegue inicializar, geralmente por:

1. **Driver NVIDIA no WSL2**: Driver não carregado ou versão incompatível
2. **GPU ocupada**: Outro processo usando a GPU
3. **Estado inconsistente**: GPU em estado de erro
4. **Incompatibilidade CUDA**: Versão do CUDA Toolkit incompatível com driver

## ✅ Solução 1: Usar CPU (RECOMENDADO - Mais Rápido)

Use o config otimizado para CPU:

```bash
# No ambiente do usuário (/home/acauanribeiro/iats/filo_priori_v8/)
python scripts/finetune_bge.py --config configs/finetune_bge_cpu.yaml
```

**Vantagens**:
- ✅ Funciona imediatamente (sem precisar corrigir GPU)
- ✅ Config já otimizado: batch_size=8, sample_size=10000
- ✅ Tempo estimado: ~2-3 horas (vs ~30 min na GPU)

**Config CPU inclui**:
- `hardware.device: "cpu"` - Força uso de CPU
- `batch_size: 8` - Otimizado para CPU (vs 96 na GPU)
- `sample_size: 10000` - Quick test (evita 100+ horas)

## 🔧 Solução 2: Tentar Corrigir GPU (Mais Lento)

### Opção A: Reiniciar WSL2

```bash
# No PowerShell/CMD do Windows (não dentro do WSL):
wsl --shutdown

# Aguardar 10 segundos, depois reabrir WSL
wsl
```

Depois tente novamente:
```bash
python scripts/finetune_bge.py --config configs/finetune_bge.yaml
```

### Opção B: Verificar Driver NVIDIA

No Windows (PowerShell):
```powershell
nvidia-smi
```

Deve mostrar a GPU. Se não mostrar, reinstale o driver NVIDIA.

### Opção C: Verificar CUDA no WSL2

```bash
# Verificar se CUDA funciona
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verificar versão CUDA
nvcc --version  # Pode não estar instalado, tudo bem
```

### Opção D: Variável de Ambiente

Tente desabilitar NVML:
```bash
export CUDA_VISIBLE_DEVICES=""  # Força CPU
python scripts/finetune_bge.py --config configs/finetune_bge.yaml
```

## 📋 Comparação de Tempos

| Configuração | Tempo (10K samples) | Tempo (Full dataset) |
|--------------|---------------------|----------------------|
| **GPU (funcional)** | ~30 minutos | ~10-15 horas |
| **CPU** | ~2-3 horas | ~100-150 horas ⚠️ |

## 🎯 Recomendação Final

**Para testar agora**:
```bash
# Use CPU config (já está pronto)
cd /home/acauanribeiro/iats/filo_priori_v8
python scripts/finetune_bge.py --config configs/finetune_bge_cpu.yaml
```

**Para produção futura**:
1. Corrija o problema de GPU no WSL2
2. Use o config original: `configs/finetune_bge.yaml`
3. Rode dataset completo na GPU (~10-15 horas)

## 📝 Modificações Aplicadas ao Código

O script `scripts/finetune_bge.py` foi atualizado (linhas 152-180):
- ✅ Respeita `hardware.device` do config
- ✅ Suporta `device: "cpu"`, `"cuda"`, ou `"auto"`
- ✅ Trata erros de GPU graciosamente
- ✅ Warning se GPU tem problemas

**Não requer alteração de código** - apenas use o config correto!
