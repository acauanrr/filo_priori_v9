# FIX: Erros no Fine-Tuning BGE

## ⚠️ ERRO MAIS RECENTE: GPU NVML Initialization Failed
```
RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED
```

**SOLUÇÃO RÁPIDA**: Use o config CPU:
```bash
python scripts/finetune_bge.py --config configs/finetune_bge_cpu.yaml
```

Ver detalhes completos em: **`FIX_GPU_NVML_ERROR.md`**

---

## Problema 1: Dependência `datasets` Faltante
```
ImportError: Please install `datasets` to use this function: `pip install datasets`.
```

## Problema 2: Learning Rate como String
```
TypeError: '<=' not supported between instances of 'float' and 'str'
```

**Causa**: YAML parseia notação científica (`3e-5`) como string ao invés de float

## ✅ Soluções Aplicadas

### Problema 1: Instalar `datasets`
Execute o seguinte comando no seu ambiente virtual ativo:

```bash
# Se você está usando o venv em /home/acauanribeiro/iats/filo_priori_v8/
cd /home/acauanribeiro/iats/filo_priori_v8
source venv/bin/activate
pip install datasets
```

**OU** simplesmente:

```bash
# Instalar diretamente no venv sem ativar
/home/acauanribeiro/iats/filo_priori_v8/venv/bin/pip install datasets
```

### Problema 2: Corrigido Automaticamente ✓

O script `scripts/finetune_bge.py` foi atualizado para converter automaticamente:
- `learning_rate` para `float` (linha 195-196)
- `triplet_margin` para `float` (linha 180)

**Não requer ação manual** - o código agora trata YAML strings corretamente.

## Explicação

### Problema 1: datasets

O `sentence-transformers` precisa da biblioteca `datasets` do Hugging Face para:
- Processar batches durante o fine-tuning
- Gerenciar checkpoints
- Salvar modelos intermediários

### Problema 2: YAML Scientific Notation

YAML parsers às vezes convertem notação científica para string:
- `learning_rate: 3e-5` → `"3e-5"` (string) ao invés de `0.00003` (float)
- PyTorch Optimizer requer `float`, não aceita string

**Solução**: Conversão explícita com `float()` no código Python

## Após Instalação

Re-execute o fine-tuning:

```bash
bash run_finetuning.sh full
```

ou

```bash
python scripts/finetune_bge.py --config configs/finetune_bge.yaml
```

## Setup Completo para Futuro

Para evitar esse problema no futuro, sempre rode:

```bash
bash setup_finetuning.sh
```

Antes de iniciar o fine-tuning. O script agora instala TODAS as dependências necessárias, incluindo `datasets`.

## Verificação

Para verificar se `datasets` foi instalado:

```bash
python -c "import datasets; print(f'✓ datasets: {datasets.__version__}')"
```
