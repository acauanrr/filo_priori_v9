#!/bin/bash
#
# Script wrapper para encoding de embeddings na GPU com truncamento
#
# Uso:
#   ./run_gpu_encoding.sh           # Com teste rápido
#   ./run_gpu_encoding.sh --skip-test  # Pular teste, ir direto
#

set -e  # Exit on error

echo "================================================================================"
echo "GPU ENCODING DE EMBEDDINGS COM TRUNCAMENTO"
echo "================================================================================"
echo ""

# Cores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check se está no diretório correto
if [ ! -f "configs/experiment.yaml" ]; then
    echo -e "${RED}✗ ERRO: Arquivo configs/experiment.yaml não encontrado${NC}"
    echo "  Execute este script do diretório raiz do projeto"
    exit 1
fi

# Check se venv está ativo
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠ AVISO: Virtual environment não ativo${NC}"
    echo "  Tentando ativar venv..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo -e "${GREEN}✓ Venv ativado${NC}"
    else
        echo -e "${RED}✗ ERRO: venv não encontrado${NC}"
        exit 1
    fi
fi

# Parse arguments
SKIP_TEST=false
if [ "$1" == "--skip-test" ]; then
    SKIP_TEST=true
fi

# PASSO 1: Teste rápido (se não --skip-test)
if [ "$SKIP_TEST" = false ]; then
    echo ""
    echo "================================================================================"
    echo "PASSO 1: TESTE RÁPIDO (10, 100, 500 samples)"
    echo "================================================================================"
    echo ""
    echo "Este teste valida que o encoding funciona corretamente na GPU."
    echo "Tempo estimado: 5 minutos"
    echo ""
    read -p "Executar teste rápido? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        python scripts/test_commit_encoding_gpu.py

        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}✓✓✓ TESTE PASSOU! ✓✓✓${NC}"
            echo ""
        else
            echo ""
            echo -e "${RED}✗✗✗ TESTE FALHOU! ✗✗✗${NC}"
            echo ""
            echo "Verifique os erros acima e tente:"
            echo "  1. Reduzir chunk_size: --chunk_size 250"
            echo "  2. Aumentar safety_margin (mais conservador)"
            echo "  3. Consultar GPU_ENCODING_WITH_TRUNCATION.md"
            exit 1
        fi
    else
        echo "Teste pulado."
    fi
fi

# PASSO 2: Encoding completo
echo ""
echo "================================================================================"
echo "PASSO 2: ENCODING COMPLETO"
echo "================================================================================"
echo ""
echo "Este processo irá:"
echo "  1. Encodar TCs na GPU (~25 min, 50621 samples)"
echo "  2. Encodar Commits na GPU com truncamento (~35 min, 50621 samples)"
echo "  3. Encodar Val set (~5 min, 6062 samples)"
echo "  4. Encodar Test set (~5 min, 6195 samples)"
echo ""
echo "Tempo estimado total: ~1 hora"
echo ""
echo "Output: cache/embeddings_precomputed.npz (~600 MB)"
echo ""

read -p "Iniciar encoding completo? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "Operação cancelada."
    exit 0
fi

echo ""
echo "Iniciando encoding..."
echo ""

# Criar diretório cache se não existir
mkdir -p cache

# Executar encoding
START_TIME=$(date +%s)

python scripts/precompute_embeddings.py \
    --config configs/experiment.yaml \
    --output cache/embeddings_precomputed.npz \
    --chunk_size 500 \
    --device cuda

ENCODING_EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

if [ $ENCODING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo -e "${GREEN}✓✓✓ ENCODING COMPLETO! ✓✓✓${NC}"
    echo "================================================================================"
    echo ""
    echo "Tempo total: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
    echo "Output: cache/embeddings_precomputed.npz"
    echo ""

    # PASSO 3: Validação
    echo "================================================================================"
    echo "PASSO 3: VALIDAÇÃO"
    echo "================================================================================"
    echo ""

    python -c "
import numpy as np
try:
    emb = np.load('cache/embeddings_precomputed.npz')
    print('✓ Arquivo carregado com sucesso')
    print('')
    print('Shapes:')
    print(f\"  Train: {emb['train_embeddings'].shape}\")
    print(f\"  Val:   {emb['val_embeddings'].shape}\")
    print(f\"  Test:  {emb['test_embeddings'].shape}\")
    print('')

    # Validar shapes
    expected = {
        'train': (50621, 3072),
        'val': (6062, 3072),
        'test': (6195, 3072)
    }

    all_ok = True
    for split, exp_shape in expected.items():
        actual_shape = emb[f'{split}_embeddings'].shape
        if actual_shape == exp_shape:
            print(f'  ✓ {split}: shape OK')
        else:
            print(f'  ✗ {split}: shape mismatch (expected {exp_shape}, got {actual_shape})')
            all_ok = False

    print('')
    if all_ok:
        print('✓✓✓ VALIDAÇÃO PASSOU! ✓✓✓')
        exit(0)
    else:
        print('✗✗✗ VALIDAÇÃO FALHOU! ✗✗✗')
        exit(1)

except Exception as e:
    print(f'✗ Erro ao carregar arquivo: {e}')
    exit(1)
"

    VALIDATION_EXIT_CODE=$?

    if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "================================================================================"
        echo -e "${GREEN}✓✓✓ SUCESSO COMPLETO! ✓✓✓${NC}"
        echo "================================================================================"
        echo ""
        echo "Embeddings pré-computados e validados!"
        echo ""
        echo "Próximos passos:"
        echo "  1. Modificar main.py para carregar embeddings pré-computados"
        echo "  2. Rodar experimentos sem re-encodar (economiza HORAS!)"
        echo ""
        echo "Exemplo:"
        echo "  # Em main.py, substituir:"
        echo "  # train_embeddings = encoder.encode(...)"
        echo "  # Por:"
        echo "  embeddings = np.load('cache/embeddings_precomputed.npz')"
        echo "  train_embeddings = embeddings['train_embeddings']"
        echo ""
    else
        echo ""
        echo -e "${RED}✗ Validação falhou${NC}"
        echo "Verifique os shapes acima"
    fi

else
    echo ""
    echo "================================================================================"
    echo -e "${RED}✗✗✗ ENCODING FALHOU! ✗✗✗${NC}"
    echo "================================================================================"
    echo ""
    echo "Verifique os erros acima."
    echo ""
    echo "Possíveis soluções:"
    echo "  1. Reduzir chunk_size:"
    echo "     python scripts/precompute_embeddings.py --chunk_size 250"
    echo ""
    echo "  2. Consultar troubleshooting:"
    echo "     cat GPU_ENCODING_WITH_TRUNCATION.md"
    echo ""
    exit 1
fi
