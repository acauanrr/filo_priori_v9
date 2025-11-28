#!/bin/bash
# =============================================================================
# EXPERIMENTO HIBRIDO: FILO-PRIORI - BEST OF BOTH WORLDS
# =============================================================================
#
# Este script executa o experimento hibrido que combina:
# - Arquitetura GATv2 comprovada (do exp_07, APFD 0.6379)
# - PhyloEncoder LITE (2 layers) - contribuicao cientifica
# - PhylogeneticDistanceKernel - contribuicao cientifica
# - PhyloRegularization (0.05) - contribuicao cientifica
# - SEM HierarchicalAttention (nao ajudou, overhead)
#
# TARGET: APFD >= 0.64 (manter ou superar exp_07)
#
# =============================================================================

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}        FILO-PRIORI V9 - EXPERIMENTO HIBRIDO (BEST OF BOTH WORLDS)         ${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""
echo -e "${YELLOW}Arquitetura:${NC}"
echo "  - Base: GATv2 (exp_07, APFD 0.6379)"
echo "  - Adicao: PhyloEncoder LITE (2 layers)"
echo "  - Adicao: PhylogeneticDistanceKernel (temperatura aprendivel)"
echo "  - Adicao: PhyloRegularization (weight=0.05)"
echo "  - Remocao: HierarchicalAttention (overhead sem ganho)"
echo ""
echo -e "${YELLOW}Target:${NC} APFD >= 0.64"
echo ""

# Verifica se esta no diretorio correto
if [ ! -f "main.py" ]; then
    echo -e "${RED}ERRO: Execute este script a partir do diretorio raiz do projeto${NC}"
    exit 1
fi

# Verifica se o arquivo de configuracao existe
CONFIG_FILE="configs/experiment_hybrid_phylogenetic.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}ERRO: Arquivo de configuracao nao encontrado: $CONFIG_FILE${NC}"
    exit 1
fi

# Cria diretorio de resultados
RESULTS_DIR="results/experiment_hybrid_phylogenetic"
mkdir -p "$RESULTS_DIR"

# Ativa ambiente virtual se existir
if [ -d "venv" ]; then
    echo -e "${GREEN}Ativando ambiente virtual...${NC}"
    source venv/bin/activate
fi

# Verifica GPU
echo -e "${YELLOW}Verificando GPU...${NC}"
python -c "import torch; print(f'CUDA disponivel: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Registra inicio
START_TIME=$(date +%s)
echo -e "${GREEN}Iniciando treinamento: $(date)${NC}"
echo ""

# Executa o experimento
echo -e "${BLUE}Executando: python main.py --config $CONFIG_FILE${NC}"
echo "============================================================================="

python main.py --config "$CONFIG_FILE" 2>&1 | tee "$RESULTS_DIR/tmux-buffer.txt"

# Registra fim
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))

echo ""
echo "============================================================================="
echo -e "${GREEN}Treinamento concluido!${NC}"
echo -e "Duracao: ${DURATION_MIN} minutos"
echo ""

# Verifica resultados
if [ -f "$RESULTS_DIR/apfd_per_build_FULL_testcsv.csv" ]; then
    echo -e "${GREEN}Resultados encontrados!${NC}"
    echo ""

    # Calcula APFD medio
    MEAN_APFD=$(awk -F',' 'NR>1 {sum+=$6; count++} END {printf "%.4f", sum/count}' "$RESULTS_DIR/apfd_per_build_FULL_testcsv.csv")

    echo "============================================================================="
    echo -e "${YELLOW}RESULTADO FINAL:${NC}"
    echo "============================================================================="
    echo -e "  Mean APFD: ${GREEN}$MEAN_APFD${NC}"
    echo ""

    # Compara com target
    TARGET=0.64
    COMPARISON=$(echo "$MEAN_APFD >= $TARGET" | bc -l)
    if [ "$COMPARISON" -eq 1 ]; then
        echo -e "  ${GREEN}✓ TARGET ATINGIDO (>= $TARGET)${NC}"
    else
        echo -e "  ${YELLOW}⚠ Abaixo do target ($TARGET)${NC}"
    fi

    # Compara com exp_07
    EXP07_APFD=0.6379
    DIFF=$(echo "$MEAN_APFD - $EXP07_APFD" | bc -l)
    if [ $(echo "$MEAN_APFD >= $EXP07_APFD" | bc -l) -eq 1 ]; then
        echo -e "  ${GREEN}✓ SUPEROU exp_07 ($EXP07_APFD) por $DIFF${NC}"
    else
        echo -e "  ${YELLOW}⚠ Abaixo do exp_07 ($EXP07_APFD) por $DIFF${NC}"
    fi
    echo ""
else
    echo -e "${RED}AVISO: Arquivo de resultados APFD nao encontrado${NC}"
fi

echo "============================================================================="
echo -e "${BLUE}Resultados salvos em: $RESULTS_DIR${NC}"
echo "============================================================================="
