#!/bin/bash
# =============================================================================
# Script de Execucao: Experimento Filogenetico - Filo-Priori V9
# =============================================================================
#
# Este script executa o experimento com a arquitetura filogenetica completa:
# - PhyloEncoder (GGNN) para processamento do Git DAG
# - Hierarchical Attention (Micro/Meso/Macro)
# - Cross-Attention Fusion
# - Combined Loss: Focal + Ranking + Phylo-Regularization
#
# Uso:
#   ./run_experiment_phylogenetic.sh
#
# O script:
# 1. Verifica o ambiente (CUDA, Python, dependencias)
# 2. Executa o treinamento com a config filogenetica
# 3. Salva logs e resultados em results/experiment_phylogenetic_v9/
#
# =============================================================================

set -e  # Exit on error

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Diretorio do projeto
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}     FILO-PRIORI V9 - EXPERIMENTO FILOGENETICO                              ${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# =============================================================================
# 1. Verificar ambiente
# =============================================================================
echo -e "${YELLOW}[1/4] Verificando ambiente...${NC}"

# Verificar Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}ERRO: Python nao encontrado!${NC}"
    exit 1
fi

echo -e "  Python: $($PYTHON_CMD --version)"

# Verificar CUDA
if $PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE="cuda"
    CUDA_INFO=$($PYTHON_CMD -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null)
    echo -e "  ${GREEN}CUDA: Disponivel${NC}"
    echo -e "  $CUDA_INFO"
else
    DEVICE="cpu"
    echo -e "  ${YELLOW}CUDA: Nao disponivel (usando CPU)${NC}"
fi

# Verificar dependencias criticas
echo -e "  Verificando dependencias..."
$PYTHON_CMD -c "import torch; import torch_geometric; import sentence_transformers" 2>/dev/null || {
    echo -e "${RED}ERRO: Dependencias nao encontradas!${NC}"
    echo -e "  Execute: pip install torch torch-geometric sentence-transformers"
    exit 1
}
echo -e "  ${GREEN}Dependencias: OK${NC}"

# =============================================================================
# 2. Verificar config e datasets
# =============================================================================
echo ""
echo -e "${YELLOW}[2/4] Verificando arquivos...${NC}"

CONFIG_FILE="configs/experiment_phylogenetic.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}ERRO: Config nao encontrado: $CONFIG_FILE${NC}"
    exit 1
fi
echo -e "  Config: ${GREEN}$CONFIG_FILE${NC}"

# Verificar datasets
if [ ! -f "datasets/train.csv" ] || [ ! -f "datasets/test.csv" ]; then
    echo -e "${RED}ERRO: Datasets nao encontrados!${NC}"
    echo -e "  Esperado: datasets/train.csv e datasets/test.csv"
    exit 1
fi
echo -e "  Datasets: ${GREEN}OK${NC}"

# =============================================================================
# 3. Criar diretorio de resultados
# =============================================================================
echo ""
echo -e "${YELLOW}[3/4] Preparando diretorios...${NC}"

RESULTS_DIR="results/experiment_phylogenetic_v9"
mkdir -p "$RESULTS_DIR"
echo -e "  Resultados: ${GREEN}$RESULTS_DIR${NC}"

# Criar arquivo de log
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$RESULTS_DIR/training_${TIMESTAMP}.log"

# =============================================================================
# 4. Executar treinamento
# =============================================================================
echo ""
echo -e "${YELLOW}[4/4] Iniciando treinamento...${NC}"
echo ""
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}  ARQUITETURA FILOGENETICA V9                                               ${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo -e "  - PhyloEncoder: GGNN com Distance Kernel"
echo -e "  - Hierarchical Attention: Micro/Meso/Macro"
echo -e "  - Loss: Focal (0.6) + Ranking (0.3) + Phylo-Reg (0.1)"
echo -e "  - Device: $DEVICE"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# Executar o treinamento
echo -e "${GREEN}Executando...${NC}"
echo ""

$PYTHON_CMD main.py \
    --config "$CONFIG_FILE" \
    --device "$DEVICE" \
    2>&1 | tee "$LOG_FILE"

# =============================================================================
# 5. Verificar resultados
# =============================================================================
echo ""
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}  EXPERIMENTO FINALIZADO                                                     ${NC}"
echo -e "${BLUE}=============================================================================${NC}"

if [ -f "$RESULTS_DIR/apfd_per_build_FULL_testcsv.csv" ]; then
    echo -e "${GREEN}Resultados salvos em: $RESULTS_DIR/${NC}"
    echo ""
    echo -e "Arquivos gerados:"
    ls -la "$RESULTS_DIR/"
    echo ""

    # Mostrar resumo APFD
    if [ -f "$RESULTS_DIR/apfd_per_build_FULL_testcsv.csv" ]; then
        echo -e "${YELLOW}Resumo APFD (277 builds):${NC}"
        # Calcular media APFD
        MEAN_APFD=$(awk -F',' 'NR>1 {sum+=$6; count++} END {printf "%.4f", sum/count}' \
            "$RESULTS_DIR/apfd_per_build_FULL_testcsv.csv" 2>/dev/null || echo "N/A")
        echo -e "  Mean APFD: ${GREEN}$MEAN_APFD${NC}"
    fi
else
    echo -e "${YELLOW}AVISO: Alguns arquivos de resultado nao foram gerados${NC}"
fi

echo ""
echo -e "Log completo: ${LOG_FILE}"
echo ""
