from llama_cpp import Llama
llm = Llama(model_path="~/llama.cpp/models/gemma-2-2b-it-Q4_K_S.gguf",
            n_ctx=32, n_threads=2, logits_all=False)
bad_words = ["*", "#"]
bad_ids = [llm.tokenize(word.encode())[0] for word in bad_words]
print(bad_ids)
