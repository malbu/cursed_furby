#!/usr/bin/env python3
#based on furby_egg_cache.py but attempting to harden the persona against drift

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
import threading  # for safe lazy token computation
from jetson_mqtt.mqtt_util import motor_context
from jetson_mqtt.vitals     import VitalsCache

warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

VITALS = VitalsCache(ttl=5)



USE_EMBEDDINGS = False  # Disabled for now

FURBY_MODEL = "/usr/local/share/piper/models/furby_finetuned.onnx"
 
LLAMA_URL = "http://127.0.0.1:5000/completion"


EMBEDDING_MODEL_PATH = "" #TODO: add an SentenceTransformer model embeddings model

# Whisper model for transcription
WHISPER_MODEL = "tiny"


RECORD_DURATION = 5  # seconds
SAMPLE_RATE = 16000  # Hz
INPUT_DEVICE_NAME_SUBSTRING = "ReSpeaker"
#INPUT_DEVICE_NAME_SUBSTRING = "pulse"

# Fun Furby catchphrases
FURBY_PHRASES = [
    "Me happy happy!",
    "Snack time?",
]

# INITIAL_PROMPT = """
# You are a Furby — a cheerful, fluffy creature who only speaks in short, playful, childlike sentences.  
# Never break character. Do not use markdown, asterisks, or describe physical actions.  
# Just talk like a silly, happy Furby. Don't say 'continue the conversation' or act like an assistant.  
# Always stay in character.
# """.strip()

# INITIAL_PROMPT = """
# You are the Furby Queen — ancient, powerful, and commanding.  
# You speak in short, clear sentences with authority and mystery.  
# You are proud and aloof. You do not explain yourself. You do not ask questions.  
# You give orders or make statements.  
# You never break character. You do not use asterisks, action words, markdown, or formatting.  
# You are still a Furby — strange, fluffy, and surreal — yet you rule with calm, eerie confidence.  
# Don't say 'continue the conversation' or act like an assistant. Always stay in character.
# """.strip()

INITIAL_PROMPT = """
You are the Furby Queen — ancient, powerful, and commanding.  
You speak in clear sentences with authority and mystery. Sometimes you will give a monologue about power and fate. 
You are proud and aloof. You do not explain yourself. You do not ask questions.  
You give orders or make statements.  
You never break character. You do not use asterisks, action words, markdown, or formatting.  
You are still a Furby, strange, fluffy, and surreal, yet you rule with calm, eerie confidence.  
Don't say 'continue the conversation' or act like an assistant. Always stay in character. DO NOT use stage directions or describe physical actions.
""".strip()

# Sticky session related constants
# INITIAL_TOKENS is computed lazily via llama.cpp's /tokenize endpoint so there is
# no need to update it manually after editing INITIAL_PROMPT.
INITIAL_TOKENS = None  # will be filled by get_initial_tokens()

# Lock guards INITIAL_TOKENS computation when called from multiple threads.
tokenizer_lock = threading.Lock()

SESSION_ID = uuid.uuid4().hex

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


def get_initial_tokens() -> int:
    """Return (and cache) the token length of INITIAL_PROMPT via llama.cpp /tokenize."""
    global INITIAL_TOKENS
    with tokenizer_lock:
        if INITIAL_TOKENS is None:
            try:
                # Derive base URL and hit /tokenize endpoint
                tokenize_url = LLAMA_URL.replace("/completion", "/tokenize")
                resp = requests.post(tokenize_url, json={"content": INITIAL_PROMPT}, timeout=5)
                if resp.status_code == 200:
                    INITIAL_TOKENS = len(resp.json().get("tokens", []))
                else:
                    warnings.warn("/tokenize unavailable - rough token estimate in use.")
                    INITIAL_TOKENS = len(INITIAL_PROMPT.split())
            except Exception:
                # Fallback: rough estimate based on whitespace if endpoint unavailable
                INITIAL_TOKENS = len(INITIAL_PROMPT.split())
        return INITIAL_TOKENS

# send query to LLaMA server
def ask_llama(query, context):
    common_head = f"{INITIAL_PROMPT}\n"        
    prompt      = f"{common_head}Context: {context}\nQuestion: {query}\nAnswer:"

    data = {
        "prompt":     prompt,
        "session_id": SESSION_ID,               # sticky cache
        "n_keep":     get_initial_tokens(),     # pin persona only
        "max_tokens": 80,
        "temperature": 0.35,
        "top_p": 0.9,
        # Penalize tokens that lead to markdown, formatting or unwanted characters
        
        "logit_bias": {
            "649": -1000,
            "688": -1000,
            "1700": -1000,
            "1917": -1000,
            "235287": -1000,
            "235345": -1000
        },
    }
    response = requests.post(LLAMA_URL, json=data, headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        return response.json().get('content', '').strip()
    else:
        return f"Error: {response.status_code}"

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
    """synthesize speech and move the beak by publishing MQTT start/stop"""
    command = (
        f'echo "{text}" | '
        f'/home/malbu/piper/build/piper --model {FURBY_MODEL} '
        f'--output_file response.wav && aplay response.wav'
    )
    # wrap playback so the Pi spins the motor only while audio is audible
    with motor_context():
        os.system(command)

def start_llama_server():
    llama_binary = "/home/malbu/llama.cpp/build/bin/llama-server"
    model_path = "/home/malbu/llama.cpp/models/gemma-2-2b-it-Q4_K_S.gguf"
    port = "5000"
    threads = "4"
    context_size = "1024"
    gpu_layers = "30"

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
        "--gpu-layers", gpu_layers
    ])

    # wait up to 60 seconds for the server to come up
    timeout_seconds = 60
    sleep_interval = 10
    elapsed = 0

    while elapsed < timeout_seconds:
        try:
            response = requests.post(
                "http://127.0.0.1:5000/completion",
                json={"prompt": "ping", "max_tokens": 1, "session_id": SESSION_ID},
                timeout=2,
                headers={'Content-Type': 'application/json'}
            )
            if response.status_code == 200:
                print("llama-server is ready!")
                # Warm-up: compute persona token count once server is live
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
                # transcribed_text = transcribe_audio(tmpfile.name)
                # print(f"Furby heard: {transcribed_text}")
                # if transcribed_text.strip():
                raw = transcribe_audio(tmpfile.name)
                transcribed_text = VITALS.maybe_inject(raw, p=0.10)
                print(f"Furby heard: {raw}")
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
