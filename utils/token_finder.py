from llama_cpp import Llama
import os, json, itertools
#script to find tokens to add to logit_bias in furby_queen.py; to penalize markdown formatting and prevent feeding it into voice model
model_path = os.path.expanduser("~/llama.cpp/models/gemma-2-2b-it-Q4_K_S.gguf")
llm = Llama(model_path=model_path, n_ctx=32, n_threads=2)

# Any literal strings you never want the model to produce
bad_words = [
    "*",  "#",  "```",  "**",
    " *",  " #",        # leading space
]

token_ids = set()
for word in bad_words:
    ids = llm.tokenize(word.encode(), add_bos=False)
    token_ids.update(ids)

# Show unique IDs sorted
print(sorted(token_ids))

# Dump JSON to paste into logit_bias
print(json.dumps({str(i): -1000 for i in sorted(token_ids)}, indent=2))
