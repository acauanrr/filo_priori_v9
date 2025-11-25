#!/bin/bash

echo "=========================================="
echo "NVML Error Fix V2 - Verification"
echo "=========================================="
echo

echo "1. Checking configuration..."
if grep -q "model_reload_frequency: 5" configs/experiment.yaml; then
    echo "   ✓ model_reload_frequency: 5 (configured)"
else
    echo "   ✗ model_reload_frequency not found in config"
    exit 1
fi

if grep -q "use_chunked_encoding: true" configs/experiment.yaml; then
    echo "   ✓ use_chunked_encoding: true"
else
    echo "   ✗ use_chunked_encoding not enabled"
    exit 1
fi

echo

echo "2. Checking existing cache..."
if [ -f "cache/embeddings_qodo/train_tc_embeddings.npy" ]; then
    SIZE=$(ls -lh cache/embeddings_qodo/train_tc_embeddings.npy | awk '{print $5}')
    echo "   ✓ TC embeddings cached (${SIZE})"
else
    echo "   ℹ TC embeddings not cached (will encode)"
fi

if [ -d "cache/embeddings_qodo/commit_chunks" ]; then
    CHUNK_COUNT=$(ls cache/embeddings_qodo/commit_chunks/ | wc -l)
    echo "   ✓ Commit chunks cached: ${CHUNK_COUNT} chunks"
else
    echo "   ℹ Commit chunks not cached (will encode from start)"
fi

echo

echo "3. Testing Python imports..."
python -c "
import sys
sys.path.insert(0, 'src')
from embeddings.qodo_encoder_chunked import QodoEncoderChunked
print('   ✓ QodoEncoderChunked import successful')
" || exit 1

echo

echo "4. Verifying code changes..."
if grep -q "_reload_model" src/embeddings/qodo_encoder_chunked.py; then
    echo "   ✓ _reload_model() method present"
else
    echo "   ✗ _reload_model() method not found"
    exit 1
fi

if grep -q "model_reload_frequency" src/embeddings/qodo_encoder_chunked.py; then
    echo "   ✓ Periodic reload logic present"
else
    echo "   ✗ Periodic reload logic not found"
    exit 1
fi

if grep -q "max_retries_per_chunk" src/embeddings/qodo_encoder_chunked.py; then
    echo "   ✓ Retry logic present"
else
    echo "   ✗ Retry logic not found"
    exit 1
fi

echo

echo "=========================================="
echo "✓ All checks passed!"
echo "=========================================="
echo
echo "Ready to run experiment:"
echo "  python main.py --config configs/experiment.yaml"
echo
echo "Expected behavior:"
echo "  - TC embeddings: Load from cache (<1 sec)"
echo "  - Commit chunks 1-2: Load from cache (<1 sec)"
echo "  - Commit chunks 3-51: Encode with periodic reloads (~17 min)"
echo "  - Model will reload at chunks: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50"
echo "  - If any chunk fails: Auto-retry with model reload"
echo
echo "Total expected time: ~18-20 minutes for embeddings"
