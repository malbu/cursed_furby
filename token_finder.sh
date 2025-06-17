#!/usr/bin/env bash
MODEL=~/llama.cpp/models/gemma-2-2b-it-Q4_K_S.gguf
declare -a BLOCKLIST=('*' '```' ' Question:' ' Answer:' $'\n*')

for s in "${BLOCKLIST[@]}"; do
  TOKENS=$(~/llama.cpp/build/bin/llama-tokenize -m "$MODEL" "$s" \
            | awk '/^[0-9]+:/ {printf "%s ", $2}')
  echo -e "'$s'  ->  $TOKENS"
done
