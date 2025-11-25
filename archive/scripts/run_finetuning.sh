#!/bin/bash
# Script Auxiliar para Fine-Tuning do BGE
# Uso: bash run_finetuning.sh [quick|full]

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}       FINE-TUNING BGE PARA DOMÍNIO DE SOFTWARE ENGINEERING           ${NC}"
echo -e "${BLUE}=======================================================================${NC}"
echo ""

# Check arguments
MODE=${1:-quick}  # Default to quick

if [[ "$MODE" != "quick" && "$MODE" != "full" ]]; then
    echo -e "${RED}Uso: bash run_finetuning.sh [quick|full]${NC}"
    echo ""
    echo "Modos:"
    echo "  quick - Teste rápido com 10K amostras (~30 minutos)"
    echo "  full  - Dataset completo (~10-15 horas)"
    exit 1
fi

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}Erro: Virtual environment não encontrado!${NC}"
    echo "Execute: python3 -m venv venv"
    exit 1
fi

echo -e "${YELLOW}Ativando virtual environment...${NC}"
source venv/bin/activate

# Check if sentence-transformers is installed
echo -e "${YELLOW}Verificando dependências...${NC}"
if ! python -c "import sentence_transformers" 2>/dev/null; then
    echo -e "${YELLOW}sentence-transformers não encontrado. Instalando...${NC}"
    bash setup_finetuning.sh
else
    echo -e "${GREEN}✓ Dependências OK${NC}"
fi

# Create necessary directories
mkdir -p logs
mkdir -p models
mkdir -p cache

# Update config based on mode
CONFIG_FILE="configs/finetune_bge.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Erro: Config file não encontrado: $CONFIG_FILE${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}=======================================================================${NC}"

if [ "$MODE" = "quick" ]; then
    echo -e "${BLUE}                    MODO: TESTE RÁPIDO (30 minutos)                    ${NC}"
    echo -e "${BLUE}=======================================================================${NC}"
    echo ""
    echo -e "${YELLOW}Configurando para 10,000 amostras...${NC}"

    # Update config for quick mode
    sed -i 's/sample_size: null/sample_size: 10000/' "$CONFIG_FILE"
    sed -i 's/sample_size: [0-9]*/sample_size: 10000/' "$CONFIG_FILE"

    echo -e "${GREEN}✓ Config atualizado: sample_size: 10000${NC}"
    echo ""
    echo -e "${YELLOW}Iniciando fine-tuning...${NC}"
    echo ""

    # Run fine-tuning in foreground for quick mode
    python scripts/finetune_bge.py --config "$CONFIG_FILE"

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}=======================================================================${NC}"
        echo -e "${GREEN}                 ✓ FINE-TUNING RÁPIDO COMPLETO!                       ${NC}"
        echo -e "${GREEN}=======================================================================${NC}"
        echo ""
        echo -e "Modelo salvo em: ${BLUE}models/finetuned_bge_v1/${NC}"
        echo ""
        echo -e "${YELLOW}Próximo passo:${NC}"
        echo "  1. Verificar modelo: ls -lh models/finetuned_bge_v1/"
        echo "  2. Rodar full: bash run_finetuning.sh full"
        echo "  3. Ou usar em V8: python main_v8.py --config configs/experiment_v8_baseline.yaml"
        echo ""
    else
        echo -e "${RED}Erro durante fine-tuning. Verifique os logs.${NC}"
        exit 1
    fi

else  # full mode
    echo -e "${BLUE}                 MODO: DATASET COMPLETO (10-15 horas)                  ${NC}"
    echo -e "${BLUE}=======================================================================${NC}"
    echo ""
    echo -e "${YELLOW}Configurando para dataset completo...${NC}"

    # Update config for full mode
    sed -i 's/sample_size: [0-9]*/sample_size: null/' "$CONFIG_FILE"
    sed -i 's/sample_size: 10000/sample_size: null/' "$CONFIG_FILE"

    echo -e "${GREEN}✓ Config atualizado: sample_size: null (dataset completo)${NC}"
    echo ""
    echo -e "${RED}ATENÇÃO: Este processo vai demorar 10-15 horas!${NC}"
    echo ""
    read -p "Continuar? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelado."
        exit 0
    fi

    echo ""
    echo -e "${YELLOW}Iniciando fine-tuning em background...${NC}"

    # Run fine-tuning in background
    nohup python scripts/finetune_bge.py --config "$CONFIG_FILE" > logs/finetune_full.log 2>&1 &
    PID=$!

    echo ""
    echo -e "${GREEN}✓ Fine-tuning iniciado em background!${NC}"
    echo ""
    echo -e "${BLUE}Informações:${NC}"
    echo "  PID: $PID"
    echo "  Log: logs/finetune_full.log"
    echo "  Tempo estimado: 10-15 horas"
    echo ""
    echo -e "${YELLOW}Comandos úteis:${NC}"
    echo ""
    echo "  # Ver progresso:"
    echo "  tail -f logs/finetune_full.log"
    echo ""
    echo "  # Monitorar GPU:"
    echo "  watch -n 1 nvidia-smi"
    echo ""
    echo "  # Verificar se está rodando:"
    echo "  ps aux | grep finetune_bge"
    echo ""
    echo "  # Matar processo (se necessário):"
    echo "  kill $PID"
    echo ""
    echo -e "${BLUE}Aguarde a conclusão (~10-15 horas)${NC}"
    echo ""
fi
