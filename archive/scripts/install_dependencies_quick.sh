#!/bin/bash
# Instalação Rápida de Dependências para Fine-Tuning
# Uso: bash install_dependencies_quick.sh

set -e

echo "======================================================================="
echo "INSTALANDO DEPENDÊNCIAS PARA FINE-TUNING"
echo "======================================================================="
echo ""

# Check if we're in the right directory
if [ ! -d "scripts" ]; then
    echo "ERRO: Execute este script do diretório raiz do projeto"
    exit 1
fi

# Activate venv if exists
if [ -d "venv" ]; then
    echo "→ Ativando virtual environment..."
    source venv/bin/activate
fi

echo "→ Instalando sentence-transformers..."
pip install sentence-transformers

echo ""
echo "→ Instalando datasets..."
pip install datasets

echo ""
echo "→ Verificando instalação..."
python -c "import sentence_transformers; print(f'✓ sentence-transformers {sentence_transformers.__version__}')"
python -c "import datasets; print(f'✓ datasets {datasets.__version__}')"

echo ""
echo "======================================================================="
echo "✓ DEPENDÊNCIAS INSTALADAS COM SUCESSO"
echo "======================================================================="
echo ""
echo "Agora execute:"
echo "  bash run_finetuning_cpu.sh"
echo ""
