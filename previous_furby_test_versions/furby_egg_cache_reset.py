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
from pathlib import Path
from sentence_transformers import SentenceTransformer
import warnings
import uuid
import threading
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")



USE_EMBEDDINGS = False  # Disabled for now

FURBY_MODEL = "/usr/local/share/piper/models/furby_finetuned.onnx"

# ------------------------------------------------------------------
# LLaMA-cpp server configuration
# ------------------------------------------------------------------

LLAMA_BASE   = "http://127.0.0.1:8080"           # base URL for all endpoints
MODEL_NAME   = "gemma-2b-it"
GPU_LAYERS   = int(os.getenv("GPU_LAYERS", 30))
MAX_CTX      = 2048                               # must match server --ctx-size

EMBEDDING_MODEL_PATH = ""  # optional SentenceTransformer model path (for future embeddings)

# Whisper model for transcription
WHISPER_MODEL = "tiny"


RECORD_DURATION = 5  # seconds
SAMPLE_RATE = 16000  # Hz
INPUT_DEVICE_NAME_SUBSTRING = "ReSpeaker"

# Fun Furby catchphrases
FURBY_PHRASES = [
    "Me happy happy!",
    "Snack time?",
]

# Triple-quoted block to avoid accidental string-glue
INITIAL_PROMPT = """
You are the Furby Queen — ancient, imperious, yet playful. You speak in short, clear sentences with authority and mystery. You are proud and aloof. You do not explain yourself. You do not ask questions. You give orders or make statements. You never break character. You do not use asterisks, actions, markdown, or formatting. Speak only in words, like a true queen. You are still a Furby — strange, fluffy, and surreal — but you rule with calm, eerie confidence. Don't say 'continue the conversation' or act like an assistant. Always stay in character.
""".strip()

# Persona handling / token counting
INITIAL_TOKENS    = None            # lazy-computed – see get_initial_tokens()
persona_injected  = False           # guards Issue 1
tokenizer_lock    = threading.Lock()

# Unique chat identifier for llama.cpp sticky cache
SESSION_ID        = uuid.uuid4().hex

# Furby lore documents
DOCS = [
    "Furbies are fluffy, playful creatures who love to talk, sing, dance, and snack. They are curious about everything!",
]

# ====================================

# load models
if USE_EMBEDDINGS:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
whisper_model = whisper.load_model(WHISPER_MODEL)

# Vector database for RAG
class VectorDatabase:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []

    def add_documents(self, docs):
        if USE_EMBEDDINGS:
            embeddings = embedding_model.encode(docs)
            self.index.add(np.array(embeddings, dtype=np.float32))
            self.documents.extend(docs)

    def search(self, query, top_k=3):
        if USE_EMBEDDINGS:
            query_embedding = embedding_model.encode([query])[0].astype(np.float32)
            distances, indices = self.index.search(np.array([query_embedding]), top_k)
            return [self.documents[i] for i in indices[0]]
        else:
            return []  # No context

db = VectorDatabase(dim=384)
db.add_documents(DOCS)

# find ReSpeaker
def find_device(device_name_substring):
    for i, device in enumerate(sd.query_devices()):
        if device_name_substring.lower() in device['name'].lower():
            return i
    raise ValueError(f"Device with name containing '{device_name_substring}' not found.")

# record
def record_audio(filename, duration=RECORD_DURATION, fs=SAMPLE_RATE):
    sd.default.device = find_device(INPUT_DEVICE_NAME_SUBSTRING)
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Done recording!")
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())

# transcribe audio 
def transcribe_audio(filename):
    return whisper_model.transcribe(filename, language="en")['text']

# --------------------------------------------------
# Helper utilities (token counting & session reset)
# --------------------------------------------------

def get_initial_tokens() -> int:
    """Return cached length of INITIAL_PROMPT. Thread-safe & lazy."""
    global INITIAL_TOKENS
    with tokenizer_lock:
        if INITIAL_TOKENS is None:
            try:
                resp = requests.post(f"{LLAMA_BASE}/tokenize", json={"content": INITIAL_PROMPT}, timeout=5)
                if resp.status_code == 200:
                    INITIAL_TOKENS = len(resp.json().get("tokens", []))
                else:
                    raise RuntimeError("tokenize endpoint error")
            except Exception:
                warnings.warn("/tokenize unavailable – rough token estimate in use.")
                INITIAL_TOKENS = len(INITIAL_PROMPT.split())
            assert INITIAL_TOKENS < MAX_CTX, "Persona exceeds context size"
    return INITIAL_TOKENS

def reset_session():
    """Generate new SESSION_ID and reclaim GPU cache (Issue 2)."""
    global SESSION_ID, persona_injected
    old_session = SESSION_ID
    # Attempt modern reset endpoint
    try:
        requests.post(f"{LLAMA_BASE}/session/{old_session}/reset", timeout=1)
    except requests.exceptions.RequestException:
        # Fallback to legacy slot erase
        try:
            requests.post(f"{LLAMA_BASE}/slots/0?action=erase", timeout=1)
        except requests.exceptions.RequestException:
            pass

    SESSION_ID = uuid.uuid4().hex
    persona_injected = False
    print(f"🔄 Session reset → {SESSION_ID[:8]}")

# ----------------------------------
# Chat-based interaction with llama-cpp
# ----------------------------------

def ask_llama(user_msg: str, context: str = "") -> str:
    """Send a single user turn to llama-cpp chat API.

    Handles persona injection only on the first call of the session (Issue 1).
    """
    global persona_injected

    # Optionally prepend retrieved context
    content = user_msg
    if context:
        content = f"Context: {context}\n\n{user_msg}"

    if not persona_injected:
        # First turn – inject persona inside the user role
        message_content = f"{INITIAL_PROMPT}\n{content}"
        msgs = [{"role": "user", "content": message_content}]
        n_keep = get_initial_tokens()  # pin only the persona
        persona_injected = True
    else:
        # Subsequent turns
        msgs = [{"role": "user", "content": content}]
        n_keep = 0

    payload = {
        "model": MODEL_NAME,
        "session_id": SESSION_ID,
        "messages": msgs,
        "n_keep": n_keep,
        "max_tokens": 80,
        "temperature": 0.7,
    }

    try:
        resp = requests.post(f"{LLAMA_BASE}/completion", json=payload, headers={"Content-Type": "application/json"})
        if resp.status_code == 200:
            return resp.json().get("content", "").strip()
    except requests.exceptions.RequestException as e:
        return f"Error contacting llama-cpp: {e}"
    return f"Error: {resp.status_code}" if 'resp' in locals() else "Error: no response"

# RAG wrapper with optional Furby-ification
def rag_ask(query):
    context = " ".join(db.search(query)) if USE_EMBEDDINGS else ""
    answer = ask_llama(query, context)
    if random.random() < 0.1:  # 10% chance to Furby-ify
        if random.random() < 0.5:
            answer = random.choice(FURBY_PHRASES) + " " + answer
        else:
            answer = answer + " " + random.choice(FURBY_PHRASES)
    return answer

# convert text to speech using Furby Piper voice
def text_to_speech(text):
    command = f'echo "{text}" | /home/malbu/piper/build/piper --model {FURBY_MODEL} --output_file response.wav && aplay response.wav'
    os.system(command)

def start_llama_server():
    llama_binary = "/home/malbu/llama.cpp/build/bin/llama-server"
    model_path = f"/home/malbu/llama.cpp/models/{MODEL_NAME}.Q4_K_M.gguf"  # example path
    port = "8080"
    threads = "4"
    context_size = str(MAX_CTX)
    gpu_layers = str(GPU_LAYERS)

    if not Path(llama_binary).exists():
        print("Error: llama-server binary not found!")
        return None

    print("Launching llama-server in background...")

    proc = subprocess.Popen([
        llama_binary,
        "-m", model_path,
        "--port", port,
        "-t", threads,
        "-c", context_size,
        "--gpu-layers", gpu_layers,
        "--chat-template", "gemma"
    ])

    # wait up to 60 seconds for the server to come up
    timeout_seconds = 60
    sleep_interval = 10
    elapsed = 0

    while elapsed < timeout_seconds:
        try:
            response = requests.post(
                f"{LLAMA_BASE}/completion",
                json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1, "session_id": SESSION_ID},
                timeout=2,
                headers={'Content-Type': 'application/json'}
            )
            if response.status_code == 200:
                print("llama-server is ready!")
                # warm up token count
                _ = get_initial_tokens()
                return proc
        except requests.exceptions.RequestException:
            pass
        print(f"Waiting for llama-server... ({elapsed}s)")
        time.sleep(sleep_interval)
        elapsed += sleep_interval

    print("Error: llama-server did not start in time.")
    proc.terminate()
    return None

def main():
    while True:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                record_audio(tmpfile.name)
                transcribed_text = transcribe_audio(tmpfile.name)
                print(f"Furby heard: {transcribed_text}")
                cleaned = transcribed_text.lower().strip()
                if "new audience" in cleaned:
                    print("Voice command detected: New audience – resetting session.")
                    text_to_speech("A new audience approaches.")
                    reset_session()
                    continue  # skip further processing this turn

                if transcribed_text.strip():
                    response = rag_ask(transcribed_text)
                    print(f"Furby says: {response}")
                    text_to_speech(response)
        except KeyboardInterrupt:
            print("\nFurby says bye bye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    llama_proc = start_llama_server()
    if not llama_proc:
        print("Furby cannot speak without its brain (llama-server)!")
        exit(1)
    try:
        main()
    finally:
        print("Shutting down llama-server...")
        llama_proc.terminate()
