from llama_cpp import Llama
import os

# Expand the user home directory (~) to an absolute path so that it works across environments
model_path = os.path.expanduser("~/llama.cpp/models/gemma-2-2b-it-Q4_K_S.gguf")
llm = Llama(model_path=model_path,
            n_ctx=32, n_threads=2, logits_all=False)
bad_words = ["*", "#","```","**"]
bad_ids = [llm.tokenize(word.encode(), add_bos=False)[0] for word in bad_words]
print(bad_ids)
