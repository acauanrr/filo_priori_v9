# Instruções Rápidas - Correção NVML/CUDA

## Problema
```
RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED
UserWarning: Can't initialize NVML
```

## Causa
1. PyTorch instalado com CUDA 12.8, mas sistema tem CUDA 12.2
2. Variáveis de ambiente configuradas tarde demais
3. NVML não acessível (não é crítico para treinar)

## Solução em 3 Passos

### PASSO 1: Aplicar Correção Automática

```bash
cd /home/acauanribeiro/iats/filo_priori_v8
source venv/bin/activate
./fix_cuda_nvml.sh
```

Este script vai:
- ✅ Reinstalar PyTorch com CUDA 12.1 (compatível com 12.2)
- ✅ Testar GPU
- ✅ Verificar que tudo está funcionando

**Tempo estimado**: 3-5 minutos

### PASSO 2: Testar Correção (Opcional mas Recomendado)

```bash
python test_gpu_embedding.py
```

Deve mostrar:
```
ALL TESTS PASSED! ✓
```

**Tempo estimado**: 1-2 minutos

### PASSO 3: Executar Experimento

```bash
./run_experiment.sh --device cuda
```

Agora deve funcionar sem erros NVML!

## O Que Foi Mudado no Código

### 1. `main.py` (linhas 16-20)
```python
# ANTES:
import os
os.environ.setdefault("PYTORCH_NO_NVML", "1")
import torch

# DEPOIS:
import os
os.environ["PYTORCH_NO_NVML"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
```

### 2. `src/embeddings/qodo_encoder.py` (linhas 7-9)
```python
# REMOVIDO (não fazia nada):
os.environ.setdefault("PYTORCH_NO_NVML", "1")
```

### 3. Novos Arquivos
- `fix_cuda_nvml.sh` - Script de correção automática
- `test_gpu_embedding.py` - Teste rápido
- `NVML_FIX_GUIDE.md` - Guia completo
- `QUICK_FIX_INSTRUCTIONS.md` - Este arquivo

## Verificação Rápida

### Antes da Correção ❌
```
PyTorch: 2.9.0+cu128
CUDA: 12.8
Status: Incompatível com sistema (CUDA 12.2)
Resultado: NVML errors, falha na geração de embeddings
```

### Depois da Correção ✅
```
PyTorch: 2.x.x+cu121
CUDA: 12.1
Status: Compatível com sistema (CUDA 12.2)
Resultado: Embeddings gerados com sucesso na GPU
```

## FAQ

**P: Por que desabilitar NVML?**
R: NVML é só para monitoramento. Não afeta o treinamento do modelo. GPU funciona 100% sem ele.

**P: Vou perder performance?**
R: Não! NVML é só para estatísticas. O treinamento usa CUDA, que continua ativo.

**P: Preciso reinstalar tudo?**
R: Não! O script só reinstala PyTorch. Todos os outros pacotes ficam intactos.

**P: E se der erro mesmo assim?**
R: Consulte o `NVML_FIX_GUIDE.md` para troubleshooting detalhado.

## Resumo Executivo

| Item | Status |
|------|--------|
| Diagnóstico | ✅ Completo |
| Correção de código | ✅ Implementada |
| Script de fix | ✅ Pronto |
| Testes | ✅ Incluídos |
| Documentação | ✅ Completa |

**Próxima ação**: Execute `./fix_cuda_nvml.sh` no servidor

---

**Se tiver qualquer dúvida, consulte o guia completo em `NVML_FIX_GUIDE.md`**
