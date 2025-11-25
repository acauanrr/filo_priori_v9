#!/usr/bin/env bash
# Quick comparison script for experiments
# Usage: ./scripts/compare_experiments_quick.sh 015 016

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <exp_id_1> <exp_id_2> [exp_id_3 ...]"
    echo "Example: $0 015 016"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              EXPERIMENT COMPARISON TOOL                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Find experiment directories
declare -A exp_dirs
for exp_id in "$@"; do
    # Try multiple possible directory patterns
    if [ -d "results/experiment_${exp_id}_optimized" ]; then
        exp_dirs[$exp_id]="results/experiment_${exp_id}_optimized"
    elif [ -d "results/experiment_${exp_id}_gatv2_rewired" ]; then
        exp_dirs[$exp_id]="results/experiment_${exp_id}_gatv2_rewired"
    elif [ -d "results/experiment_${exp_id}_best_practices" ]; then
        exp_dirs[$exp_id]="results/experiment_${exp_id}_best_practices"
    elif [ -d "results/experiment_${exp_id}" ]; then
        exp_dirs[$exp_id]="results/experiment_${exp_id}"
    else
        echo "⚠️  Warning: Experiment $exp_id not found, skipping..."
        continue
    fi
done

# Check if any experiments were found
if [ ${#exp_dirs[@]} -eq 0 ]; then
    echo "❌ No experiments found!"
    exit 1
fi

echo "📊 Found ${#exp_dirs[@]} experiment(s):"
for exp_id in "${!exp_dirs[@]}"; do
    echo "   • Exp $exp_id: ${exp_dirs[$exp_id]}"
done
echo ""

# Extract metrics for each experiment
echo "═══════════════════════════════════════════════════════════════"
echo "                    TEST METRICS COMPARISON"
echo "═══════════════════════════════════════════════════════════════"
printf "%-10s %-12s %-12s %-12s %-12s\n" "Exp ID" "Accuracy" "F1 Macro" "F1 Weighted" "Mean APFD"
echo "───────────────────────────────────────────────────────────────"

for exp_id in $(echo "${!exp_dirs[@]}" | tr ' ' '\n' | sort); do
    exp_dir="${exp_dirs[$exp_id]}"
    log_file="$exp_dir/tmux-buffer.txt"
    apfd_file="$exp_dir/apfd_per_build_FULL_testcsv.csv"

    if [ -f "$log_file" ]; then
        # Extract accuracy
        accuracy=$(grep "Final Test Accuracy:" "$log_file" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")

        # Extract F1 Macro
        f1_macro=$(grep "Final Test F1 (Macro):" "$log_file" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")

        # Extract F1 Weighted
        f1_weighted=$(grep "Final Test F1 (Weighted):" "$log_file" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")

        # Calculate mean APFD
        if [ -f "$apfd_file" ]; then
            mean_apfd=$(awk -F',' 'NR>1 {sum+=$6; count++} END {if (count>0) printf "%.4f", sum/count; else print "N/A"}' "$apfd_file")
        else
            mean_apfd="N/A"
        fi

        printf "%-10s %-12s %-12s %-12s %-12s\n" "$exp_id" "$accuracy" "$f1_macro" "$f1_weighted" "$mean_apfd"
    else
        printf "%-10s %-12s %-12s %-12s %-12s\n" "$exp_id" "N/A" "N/A" "N/A" "N/A"
        echo "   ⚠️  Log file not found: $log_file"
    fi
done

echo "═══════════════════════════════════════════════════════════════"
echo ""

# APFD Distribution comparison
echo "═══════════════════════════════════════════════════════════════"
echo "                   APFD DISTRIBUTION (277 builds)"
echo "═══════════════════════════════════════════════════════════════"

for exp_id in $(echo "${!exp_dirs[@]}" | tr ' ' '\n' | sort); do
    exp_dir="${exp_dirs[$exp_id]}"
    apfd_file="$exp_dir/apfd_per_build_FULL_testcsv.csv"

    echo "Experiment $exp_id:"
    if [ -f "$apfd_file" ]; then
        awk -F',' 'NR>1 {
            if ($6 >= 0.9) count90++;
            if ($6 >= 0.8) count80++;
            if ($6 >= 0.7) count70++;
            if ($6 >= 0.5) count50++;
            if ($6 < 0.5) count_below50++;
            total++;
        } END {
            if (total > 0) {
                printf "   • APFD ≥ 0.9: %3d (%5.1f%%)\n", count90, count90*100/total;
                printf "   • APFD ≥ 0.8: %3d (%5.1f%%)\n", count80, count80*100/total;
                printf "   • APFD ≥ 0.7: %3d (%5.1f%%)\n", count70, count70*100/total;
                printf "   • APFD ≥ 0.5: %3d (%5.1f%%)\n", count50, count50*100/total;
                printf "   • APFD < 0.5: %3d (%5.1f%%)\n", count_below50, count_below50*100/total;
            } else {
                print "   No data available";
            }
        }' "$apfd_file"
    else
        echo "   ⚠️  APFD file not found"
    fi
    echo ""
done

echo "═══════════════════════════════════════════════════════════════"
echo ""

# Configuration comparison
echo "═══════════════════════════════════════════════════════════════"
echo "                 KEY CONFIGURATION DIFFERENCES"
echo "═══════════════════════════════════════════════════════════════"

for exp_id in $(echo "${!exp_dirs[@]}" | tr ' ' '\n' | sort); do
    exp_dir="${exp_dirs[$exp_id]}"
    config_file="$exp_dir/config_used.yaml"

    echo "Experiment $exp_id:"
    if [ -f "$config_file" ]; then
        # Extract key configs
        focal_alpha=$(grep "focal_alpha:" "$config_file" | head -1 | awk '{print $2, $3}' || echo "N/A")
        use_class_weights=$(grep "use_class_weights:" "$config_file" | head -1 | awk '{print $2}' || echo "N/A")
        label_smoothing=$(grep "label_smoothing:" "$config_file" | head -1 | awk '{print $2}' || echo "N/A")
        layer_type=$(grep "layer_type:" "$config_file" | head -1 | awk '{print $2}' | tr -d '"' || echo "N/A")
        use_rewired=$(grep "use_rewired_graph:" "$config_file" | head -1 | awk '{print $2}' || echo "N/A")

        echo "   • focal_alpha: $focal_alpha"
        echo "   • use_class_weights: $use_class_weights"
        echo "   • label_smoothing: $label_smoothing"
        echo "   • GNN layer_type: $layer_type"
        echo "   • use_rewired_graph: $use_rewired"

        # Check rewiring config if exists
        rewiring_summary="$exp_dir/rewiring/rewiring_summary.yaml"
        if [ -f "$rewiring_summary" ]; then
            k=$(grep "^  k:" "$rewiring_summary" | awk '{print $2}' || echo "N/A")
            keep_ratio=$(grep "keep_original_ratio:" "$rewiring_summary" | awk '{print $2}' || echo "N/A")
            echo "   • Rewiring k: $k"
            echo "   • Rewiring keep_ratio: $keep_ratio"
        fi
    else
        echo "   ⚠️  Config file not found"
    fi
    echo ""
done

echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "✅ Comparison complete!"
echo ""
echo "💡 Tip: For detailed analysis, check:"
echo "   • Full logs: results/experiment_XXX/tmux-buffer.txt"
echo "   • Confusion matrix: results/experiment_XXX/confusion_matrix.png"
echo "   • Rewiring stats: results/experiment_XXX/rewiring/rewiring_summary.yaml"
