#!/bin/bash
# integrated into furby_queen.py
# start llama-server on 127.0.0.1:5000

MODEL_PATH="../models/gemma-2-2b-it-Q4_K_S.gguf"
PORT=5000
THREADS=4
CONTEXT_SIZE=1024
GPU_LAYERS=30

echo "Starting llama-server..."
echo "Model: $MODEL_PATH"
echo "Listening on port: $PORT"

./bin/llama-server \
  -m "$MODEL_PATH" \
  --port "$PORT" \
  -t "$THREADS" \
  -c "$CONTEXT_SIZE" \
  --gpu-layers "$GPU_LAYERS"
