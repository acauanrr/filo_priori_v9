#!/bin/bash
# Monitor script for Experiment 05

LOG_FILE="results/experiment_05_expanded_features/tmux-buffer.txt"

echo "=== Experiment 05 Monitor ==="
echo "Time: $(date)"
echo ""

# Check if process is running
if pgrep -f "experiment_05_expanded_features" > /dev/null; then
    echo "✅ Status: RUNNING"

    # Get PID and resource usage
    PID=$(pgrep -f "experiment_05_expanded_features" | head -1)
    echo "   PID: $PID"
    ps -p $PID -o %cpu,%mem,etime --no-headers | awk '{print "   CPU: "$1"% | MEM: "$2"% | Runtime: "$3}'
    echo ""

    # GPU usage
    echo "🎮 GPU Usage:"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        awk '{print "   Memory: "$1" / "$2" MB | Utilization: "$3"%"}'
    echo ""

    # Check current phase
    echo "📍 Current Phase:"
    if grep -q "GENERATING EMBEDDINGS" "$LOG_FILE" 2>/dev/null; then
        # Check embedding progress
        LAST_PROGRESS=$(tail -20 "$LOG_FILE" | grep -oP '\d+%' | tail -1)
        LAST_PHASE=$(tail -20 "$LOG_FILE" | grep -oP '(Train|Test) (TCs|Commits):' | tail -1)
        if [ -n "$LAST_PROGRESS" ]; then
            echo "   🔄 Embedding Generation: $LAST_PHASE $LAST_PROGRESS"
        else
            echo "   🔄 Embedding Generation: In progress..."
        fi
    elif grep -q "Extracting structural features" "$LOG_FILE" 2>/dev/null; then
        echo "   🔄 Structural Feature Extraction (29 features V2)"
    elif grep -q "STEP 3: MODEL TRAINING" "$LOG_FILE" 2>/dev/null; then
        # Get epoch info
        EPOCH=$(tail -50 "$LOG_FILE" | grep -oP 'Epoch \d+/\d+' | tail -1)
        if [ -n "$EPOCH" ]; then
            echo "   🔄 Model Training: $EPOCH"
            # Show latest metrics
            tail -10 "$LOG_FILE" | grep -E "(train_loss|val_f1|val_accuracy)" | tail -3
        else
            echo "   🔄 Model Training: Starting..."
        fi
    elif grep -q "STEP 4: TEST EVALUATION" "$LOG_FILE" 2>/dev/null; then
        echo "   🔄 Test Evaluation: Computing APFD..."
    elif grep -q "Final Test APFD" "$LOG_FILE" 2>/dev/null; then
        echo "   ✅ COMPLETED!"
        echo ""
        echo "📊 Final Results:"
        grep -E "Final Test (APFD|Accuracy|F1)" "$LOG_FILE" | tail -5
    else
        echo "   🔄 Data Preparation"
    fi

else
    echo "❌ Status: NOT RUNNING"

    # Check if completed
    if [ -f "$LOG_FILE" ] && grep -q "Final Test APFD" "$LOG_FILE"; then
        echo ""
        echo "✅ Experiment COMPLETED!"
        echo ""
        echo "📊 Final Results:"
        grep -E "Final Test (APFD|Accuracy|F1)" "$LOG_FILE" | tail -5
    else
        echo ""
        echo "⚠️  Experiment may have crashed. Check logs:"
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "Last 10 lines of log:"
            tail -10 "$LOG_FILE"
        else
            echo "   No log file found"
        fi
    fi
fi

echo ""
echo "============================================"
