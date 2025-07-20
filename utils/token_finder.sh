#!/usr/bin/env bash
# couldn't get this to work; use python script instead
MODEL=~/llama.cpp/models/gemma-2-2b-it-Q4_K_S.gguf
declare -a BLOCKLIST=('*' '```' ' Question:' ' Answer:' $'\n*')

for s in "${BLOCKLIST[@]}"; do
  TOKENS=$(printf '%s\n' "$s" | ~/llama.cpp/build/bin/llama-tokenize -m "$MODEL" --stdin \
            | awk '/->/ {printf "%s ", $NF}')
  echo -e "'$s'  ->  $TOKENS"
done
