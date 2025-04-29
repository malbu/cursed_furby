#!/usr/bin/env python3

import subprocess
from pathlib import Path


PIPER_BIN = "piper"
MODEL_PATH = "./cursed_furby/furby_finetuned.onnx"
OUTPUT_DIR = "./cursed_furby/furby_outputs"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

sentences = [
    "Me see you!",
    "Furby love you!",
    "Time to play!",
    "Party time!",
    "Chaos reigns!",
]

for i, sentence in enumerate(sentences):
    output_file = Path(OUTPUT_DIR) / f"furby_output_{i}.wav"
    print(f"Generating: {output_file}")

    subprocess.run(
        [
            PIPER_BIN,
            "-m", str(MODEL_PATH),
            "--output_file", str(output_file),
        ],
        input=sentence.encode("utf-8"),
        check=True
    )

print("Done generating Furby WAV files!")
