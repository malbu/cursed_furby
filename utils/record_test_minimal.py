#!/usr/bin/env python3
"""
minimal capture script using Respeaker
capture ONE mono channel at 16 kHz
no AGC / AEC / noise suppression
Usage:
    python record_furby_style.py  out.wav  [duration_seconds]
"""
import sounddevice as sd
import soundfile as sf
import sys

FS = 16_000                     # sample rate Hz
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUTFILE  = sys.argv[1] if len(sys.argv) > 1 else "furby_style.wav"


def find_respeaker() -> int:
    """Return device index of the first input with 'ReSpeaker' in its name."""
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and "ReSpeaker" in dev["name"]:
            return idx
    raise RuntimeError("No ReSpeaker microphone found.")


def main() -> None:
    idx = find_respeaker()
    sd.default.device = (idx, None)           # (input, output)
    print(f"Recording {DURATION}s from device #{idx}: {sd.query_devices(idx)['name']}")

    audio = sd.rec(int(DURATION * FS),
                   samplerate=FS,
                   channels=1,
                   dtype="int16")
    sd.wait()

    sf.write(OUTFILE, audio, FS)
    print(f"Saved {OUTFILE}")


if __name__ == "__main__":
    main()
