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


# Configuration 

USE_EMBEDDINGS = False  # Disabled for now; set True to enable RAG embeddings

FURBY_MODEL = "/usr/local/share/piper/models/furby_finetuned.onnx"
LLAMA_URL = "http://127.0.0.1:5000/completion"

EMBEDDING_MODEL_PATH = ""  
WHISPER_MODEL = "tiny"

RECORD_DURATION = 5  # seconds
SAMPLE_RATE = 16000  # Hz
INPUT_DEVICE_NAME_SUBSTRING = "ReSpeaker"

# Fun Furby catchphrases
FURBY_PHRASES = [
    "Me happy happy!",
    "Snack time?",
]

# Persona / system prompt (sent only once thanks to sticky session)
INITIAL_PROMPT = (
    "You are a Furby — a cheerful, fluffy creature who only speaks in short, playful, childlike sentences. "
    "Never break character. Do not use markdown, asterisks, or describe physical actions. "
    "Just talk like a silly, happy Furby. Don't say 'continue the conversation' or act like an assistant. "
    "Always stay in character."
)

# Documents for RAG
DOCS = [
    "Furbies are fluffy, playful creatures who love to talk, sing, dance, and snack. They are curious about everything!",
]

# Model loading

if USE_EMBEDDINGS:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
whisper_model = whisper.load_model(WHISPER_MODEL)

#  Vector DB 

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
            return []

# Instantiate and populate DB
DB_DIM = 384  # default dimension for most MiniLM-size ST models
vector_db = VectorDatabase(dim=DB_DIM)
vector_db.add_documents(DOCS)

#Audio helpers

def find_device(device_name_substring):
    for i, device in enumerate(sd.query_devices()):
        if device_name_substring.lower() in device['name'].lower():
            return i
    raise ValueError(f"Device with name containing '{device_name_substring}' not found.")


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


def transcribe_audio(filename):
    return whisper_model.transcribe(filename, language="en")['text']

#  Sticky-session LLama wrapper

SESSION_ID = uuid.uuid4().hex  # unique conversation ID
_first_prompt_sent = False      # module-level flag


def _clean_output(text: str) -> str:
    """Strip simple markdown or role-play action markers from LLM output."""
    # remove *action* blocks
    text = re.sub(r"\*[^\*]+\*", '', text)
    # strip markdown special characters
    text = re.sub(r"[\_\~`#>\[\]]", '', text)
    # collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def ask_llama(user_query: str, context: str):
    """Send a prompt to llama-server using a sticky session."""
    global _first_prompt_sent

    # Build the prompt depending on whether we've already sent the persona
    prompt_parts = []
    if not _first_prompt_sent:
        prompt_parts.append(INITIAL_PROMPT)
    if context:
        prompt_parts.append(f"Context: {context}")
    prompt_parts.append(f"Human: {user_query}")
    prompt_parts.append("Furby:")
    prompt = "\n".join(prompt_parts)

    data = {
        "prompt": prompt,
        "max_tokens": 80,
        "temperature": 0.7,
        "session_id": SESSION_ID,
        "n_keep": -1, 
    }

    try:
        response = requests.post(LLAMA_URL, json=data, headers={'Content-Type': 'application/json'})
        _first_prompt_sent = True  # Mark that we've sent the persona
        if response.status_code == 200:
            return response.json().get('content', '').strip()
        else:
            return f"Error: {response.status_code}"
    except requests.exceptions.RequestException as exc:
        return f"Error contacting llama-server: {exc}"


# RAG wrapper with Furby-ification 

def rag_ask(query):
    context = " ".join(vector_db.search(query)) if USE_EMBEDDINGS else ""
    answer = ask_llama(query, context)
    answer = _clean_output(answer)  # Safety filter

    # 10% chance to add a Furby catchphrase
    if random.random() < 0.1 and answer:
        if random.random() < 0.5:
            answer = random.choice(FURBY_PHRASES) + " " + answer
        else:
            answer = answer + " " + random.choice(FURBY_PHRASES)
    return answer



def text_to_speech(text):
    command = (
        f'echo "{text}" | /home/malbu/piper/build/piper '
        f'--model {FURBY_MODEL} --output_file response.wav && aplay response.wav'
    )
    os.system(command)

# LLama-server launch helper
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

    # Wait until the server is responsive (up to 60 s)
    timeout_seconds = 60
    sleep_interval = 10
    elapsed = 0

    while elapsed < timeout_seconds:
        try:
            ping = requests.post(
                LLAMA_URL,
                json={"prompt": "ping", "max_tokens": 1, "session_id": SESSION_ID},
                timeout=2,
                headers={'Content-Type': 'application/json'}
            )
            if ping.status_code == 200:
                print("llama-server is ready!")
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