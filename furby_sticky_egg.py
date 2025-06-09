#!/usr/bin/env python3


import os
import tempfile
import wave
import whisper
import requests
import sounddevice as sd
import numpy as np
import faiss
import random
import subprocess
import time
import re
import uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

USE_EMBEDDINGS = False  # set True to enable RAG embeddings

FURBY_MODEL = "/usr/local/share/piper/models/furby_finetuned.onnx"
LLAMA_URL = "http://127.0.0.1:5000"  

EMBEDDING_MODEL_PATH = ""
WHISPER_MODEL = "tiny"

RECORD_DURATION = 5  # seconds
SAMPLE_RATE = 16000  # Hz
INPUT_DEVICE_NAME_SUBSTRING = "ReSpeaker"

# Fun Furby catch‑phrases
FURBY_PHRASES = [
    "Me happy happy!",
    "Snack time?",
]

# Persona text 
PERSONA_TEXT = (
    "You are a Furby — a cheerful, fluffy creature who only speaks in short, playful, childlike sentences. "
    "Never break character. Do not use markdown, asterisks, or describe physical actions. "
    "Just talk like a silly, happy Furby. Don't say 'continue the conversation' or act like an assistant. "
    "Always stay in character."
)
PERSONA_TOKENS = 100  

# Documents for RAG 
DOCS = [
    "Furbies are fluffy, playful creatures who love to talk, sing, dance, and snack. They are curious about everything!",
]

# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

if USE_EMBEDDINGS:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
whisper_model = whisper.load_model(WHISPER_MODEL)

# Vector DB wrapper --------------------------------------------------

class VectorDatabase:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.documents: list[str] = []

    def add_documents(self, docs: list[str]):
        if USE_EMBEDDINGS:
            embeddings = embedding_model.encode(docs)
            self.index.add(np.array(embeddings, dtype=np.float32))
            self.documents.extend(docs)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        if USE_EMBEDDINGS:
            q_emb = embedding_model.encode([query])[0].astype(np.float32)
            dists, idxs = self.index.search(np.array([q_emb]), top_k)
            return [self.documents[i] for i in idxs[0]]
        return []

DB_DIM = 384  # MiniLM default
vector_db = VectorDatabase(DB_DIM)
vector_db.add_documents(DOCS)

# ------------------------------------------------------------------
# Audio helpers
# ------------------------------------------------------------------

def find_device(name_sub: str) -> int:
    for i, dev in enumerate(sd.query_devices()):
        if name_sub.lower() in dev['name'].lower():
            return i
    raise ValueError(f"Audio device containing '{name_sub}' not found")

def record_audio(path: str, duration: int = RECORD_DURATION, fs: int = SAMPLE_RATE):
    sd.default.device = find_device(INPUT_DEVICE_NAME_SUBSTRING)
    print("Recording…")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Done recording!")
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())


def transcribe_audio(path: str) -> str:
    return whisper_model.transcribe(path, language="en")['text']

# ------------------------------------------------------------------
# Sticky‑session wrapper
# ------------------------------------------------------------------

SESSION_ID = uuid.uuid4().hex  # unique chat id
_last_user: str | None = None
_last_assistant: str | None = None

# Clean‑up helper -----------------------------------------------------

def _clean_output(text: str) -> str:
    text = re.sub(r'\*[^\*]+\*', '', text)  # remove *actions*
    text = re.sub(r"[\_\~`#>\[\]]", '', text)  # strip markdown
    text = re.sub(r"\s{2,}", ' ', text)
    return text.strip()


def ask_llama(user_msg: str, context: str) -> str:
    """Send prompt to llama‑server, maintaining sticky KV cache."""
    global _last_user, _last_assistant

    prompt_parts: list[str] = [PERSONA_TEXT]

    # replay previous turn so the prefix matches the cache
    if _last_user:
        prompt_parts.append(_last_user)
    if _last_assistant:
        prompt_parts.append(_last_assistant)

    if context:
        prompt_parts.append(f"Context: {context}")

    prompt_parts.append(f"Human: {user_msg}")
    prompt_parts.append("Furby:")
    prompt = "\n".join(prompt_parts)

    data = {
        "prompt": prompt,
        "session_id": SESSION_ID,
        "n_keep": PERSONA_TOKENS,  # pin persona forever
        "max_tokens": 80,
        "temperature": 0.7,
    }

    try:
        r = requests.post(f"{LLAMA_URL}/completion", json=data, timeout=120)
        r.raise_for_status()
        raw_answer = r.json().get("content", "")
    except requests.exceptions.RequestException as exc:
        return f"Error contacting llama‑server: {exc}"

    answer = _clean_output(raw_answer)

    # update rolling memory after cleaning so cache prefix matches
    _last_user = f"Human: {user_msg}"
    _last_assistant = f"Furby: {answer}" if answer else None

    return answer

# ------------------------------------------------------------------
# RAG wrapper & Furby‑ification
# ------------------------------------------------------------------

def rag_ask(query: str) -> str:
    context = " ".join(vector_db.search(query)) if USE_EMBEDDINGS else ""
    response = ask_llama(query, context)

    if random.random() < 0.1 and response:
        if random.random() < 0.5:
            response = random.choice(FURBY_PHRASES) + " " + response
        else:
            response = response + " " + random.choice(FURBY_PHRASES)
    return response

# ------------------------------------------------------------------
# TTS helper (quote‑safe)
# ------------------------------------------------------------------

def text_to_speech(text: str):
    # write to temp file to avoid shell‑quote issues
    with tempfile.NamedTemporaryFile("w", delete=False) as tf:
        tf.write(text)
        tf_path = tf.name
    os.system(
        f"cat {tf_path} | /home/malbu/piper/build/piper --model {FURBY_MODEL} --output_file response.wav && aplay response.wav"
    )
    os.unlink(tf_path)

# ------------------------------------------------------------------
# Llama‑server launch helper
# ------------------------------------------------------------------

def start_llama_server():
    llama_bin = "/home/malbu/llama.cpp/build/bin/llama-server"
    model_path = "/home/malbu/llama.cpp/models/gemma-2-2b-it-Q4_K_S.gguf"
    if not Path(llama_bin).exists():
        print("Error: llama-server binary not found!")
        return None

    proc = subprocess.Popen([
        llama_bin,
        "-m", model_path,
        "--port", "5000",
        "-t", "4",
        "-c", "1024",
        "--gpu-layers", "30",
    ])

    # wait until responsive
    for waited in range(0, 60, 5):
        try:
            ping = requests.post(
                f"{LLAMA_URL}/completion",
                json={"prompt": "ping", "max_tokens": 1, "session_id": SESSION_ID},
                timeout=2,
            )
            if ping.status_code == 200:
                print("llama‑server is ready!")
                return proc
        except requests.exceptions.RequestException:
            pass
        print(f"Waiting for llama-server… ({waited}s)")
        time.sleep(5)

    print("Error: llama-server did not start in time.")
    proc.terminate()
    return None

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def main():
    while True:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                record_audio(tmp.name)
                heard = transcribe_audio(tmp.name)
                print(f"Furby heard: {heard}")
                if heard.strip():
                    reply = rag_ask(heard)
                    print(f"Furby says: {reply}")
                    text_to_speech(reply)
        except KeyboardInterrupt:
            print("\nFurby says bye bye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    llama_proc = start_llama_server()
    if not llama_proc:
        exit(1)
    try:
        main()
    finally:
        print("Shutting down llama-server…")
        llama_proc.terminate()
