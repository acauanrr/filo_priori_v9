#!/bin/bash
#
# Executar Experimento 04b: Focal Loss APENAS
#
# Este experimento testa Focal Loss como alternativa a Weighted CE (04a)
#
# Objetivo:
#   - Comparar Weighted CE vs Focal Loss
#   - Verificar se Focal melhora Recall Not-Pass
#   - Meta: F1 Macro > 0.30, Recall Not-Pass > 0.15
#
# Configuração:
#   - Loss: Focal Loss (alpha=0.5, gamma=2.0)
#   - SEM class weights (diferente de 04a)
#   - SEM balanced sampling
#   - Modelo simplificado (GAT 1 layer, 2 heads)
#   - Graph: Multi-edge (co-failure + co-success + semantic)
#

set -e  # Exit on error

echo "========================================================================"
echo "EXPERIMENTO 04b: FOCAL LOSS APENAS"
echo "========================================================================"
echo ""
echo "Comparação: Weighted CE (04a) vs Focal Loss (04b)"
echo ""
echo "Config: configs/experiment_04b_focal_only.yaml"
echo "Output: results/experiment_04b_focal_only/"
echo ""

# Verificar se ambiente virtual está ativado
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Ambiente virtual não ativado. Ativando..."
    source venv/bin/activate
fi

# Verificar GPU
echo "Verificando GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "⚠️  GPU não detectada"
echo ""

# Limpar cache de graph (usar fresh graph build)
echo "Limpando cache de graph..."
if [ -f "cache/multi_edge_graph.pkl" ]; then
    rm cache/multi_edge_graph.pkl
    echo "✅ Cache de graph removido"
else
    echo "ℹ️  Cache de graph não existe"
fi
echo ""

# Confirmar execução
echo "========================================================================"
echo "PRONTO PARA EXECUTAR"
echo "========================================================================"
echo ""
echo "Este experimento irá:"
echo "  1. Treinar modelo com Focal Loss (alpha=0.5, gamma=2.0)"
echo "  2. Comparar resultados com 04a (Weighted CE)"
echo "  3. Verificar se Recall Not-Pass melhora"
echo ""
echo "Tempo estimado: 2-3 horas"
echo ""

read -p "Continuar? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Execução cancelada pelo usuário"
    exit 1
fi

echo ""
echo "========================================================================"
echo "INICIANDO EXPERIMENTO 04b"
echo "========================================================================"
echo ""

# Executar experimento
./venv/bin/python main.py --config configs/experiment_04b_focal_only.yaml

# Verificar se completou com sucesso
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ EXPERIMENTO 04b COMPLETADO COM SUCESSO!"
    echo "========================================================================"
    echo ""
    echo "Resultados salvos em: results/experiment_04b_focal_only/"
    echo ""

    # Mostrar métricas principais (se arquivo existe)
    if [ -f "results/experiment_04b_focal_only/tmux-buffer.txt" ]; then
        echo "Métricas principais:"
        grep -E "Best Val F1|Final Test.*F1.*Macro|Final Test.*Accuracy|Mean APFD" results/experiment_04b_focal_only/tmux-buffer.txt | tail -5
    fi

    echo ""
    echo "Próximos passos:"
    echo "  1. Analisar resultados: cat ANALYSIS_EXPERIMENT_04b.md"
    echo "  2. Comparar com 04a: grep 'F1 Macro' results/experiment_04*/tmux-buffer.txt"
    echo "  3. Verificar APFD: grep 'Mean APFD' results/experiment_04*/tmux-buffer.txt"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo "❌ EXPERIMENTO 04b FALHOU"
    echo "========================================================================"
    echo ""
    echo "Verifique os logs em: results/experiment_04b_focal_only/tmux-buffer.txt"
    echo ""
    exit 1
fi
