#!/bin/bash
# Script para FORÇAR execução do fine-tuning em CPU
# Uso: bash run_finetuning_cpu.sh

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}       FINE-TUNING BGE - MODO CPU (100% GARANTIDO)                     ${NC}"
echo -e "${BLUE}=======================================================================${NC}"
echo ""

# CRITICAL: Disable CUDA completely at OS level
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=""

echo -e "${YELLOW}→ CUDA desabilitado via variáveis de ambiente${NC}"
echo -e "${YELLOW}→ CUDA_VISIBLE_DEVICES=''${NC}"
echo ""

# Check if config exists
CONFIG_FILE="configs/finetune_bge_cpu.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Erro: Config file não encontrado: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Config file encontrado: $CONFIG_FILE${NC}"
echo ""

# Show config summary
echo -e "${BLUE}Configuração:${NC}"
echo "  - Device: CPU (forçado)"
echo "  - Batch size: 8 (otimizado para CPU)"
echo "  - Sample size: 10,000 (teste rápido)"
echo "  - Tempo estimado: 2-3 horas"
echo ""

# Confirm execution
echo -e "${YELLOW}Este processo vai demorar ~2-3 horas.${NC}"
read -p "Continuar? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelado."
    exit 0
fi

echo ""
echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}                    INICIANDO FINE-TUNING                              ${NC}"
echo -e "${BLUE}=======================================================================${NC}"
echo ""

# Create log directory
mkdir -p logs

# Run with CUDA disabled
python scripts/finetune_bge.py --config "$CONFIG_FILE" 2>&1 | tee logs/finetune_cpu.log

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}=======================================================================${NC}"
    echo -e "${GREEN}                 ✓ FINE-TUNING COMPLETO!                              ${NC}"
    echo -e "${GREEN}=======================================================================${NC}"
    echo ""
    echo -e "Modelo salvo em: ${BLUE}models/finetuned_bge_v1/${NC}"
    echo ""
    echo -e "${YELLOW}Próximo passo:${NC}"
    echo "  1. Verificar modelo: ls -lh models/finetuned_bge_v1/"
    echo "  2. Usar em V8: python main_v8.py --config configs/experiment_v8_baseline.yaml"
    echo ""
else
    echo -e "${RED}=======================================================================${NC}"
    echo -e "${RED}                 ✗ ERRO DURANTE FINE-TUNING                            ${NC}"
    echo -e "${RED}=======================================================================${NC}"
    echo ""
    echo "Ver log completo: logs/finetune_cpu.log"
    echo ""
    exit 1
fi
