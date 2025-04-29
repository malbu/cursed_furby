#!/usr/bin/env python3

import os
import subprocess

# === Configure ===
input_dir = "./FurbSounds"
output_dir = "./furb_sounds_normalized"

# === Ensure output dir exists ===
os.makedirs(output_dir, exist_ok=True)

# === Process each .wav ===
for filename in sorted(os.listdir(input_dir)):
    if filename.lower().endswith(".wav"):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        cmd = [
            "ffmpeg-normalize", in_path,
            "--target-level", "-16",
            "--loudness-range-target", "11",
            "--true-peak", "-1.5",
            "--sample-rate", "22050",
            "--audio-codec", "pcm_s16le",
            "--output-format", "wav",
            "-o", out_path,
            "--force"
        ]

        print(f"Normalizing: {filename}")
        subprocess.run(cmd, check=True)
