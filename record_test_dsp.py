#!/usr/bin/env python3
"""

  • enables ReSpeaker AEC/NS/AGC via tuning.py
  • records 6 channels at 16 kHz
  • keeps raw channel 0 only

"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import sys

FS        = 16_000
DURATION  = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUTFILE   = sys.argv[1] if len(sys.argv) > 1 else "mirror_style.wav"
CHANNELS  = 6


def find_respeaker():
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] >= CHANNELS and "ReSpeaker" in dev["name"]:
            return idx, dev["name"]
    raise RuntimeError("No 6-channel ReSpeaker found.")


def enable_dsp(dev_name: str):
    try:
        from tuning import Tuning
        t = Tuning(dev_name)
        t.aec_enable(True)
        t.ns_enable(True)
        t.agc_enable(True)
        t.save()
        print("DSP enabled (AEC/NS/AGC).")
    except Exception as e:
        print(f"WARNING: DSP not enabled – {e}")


def main():
    idx, name = find_respeaker()
    print(f"Recording {DURATION}s from #{idx}: {name}")
    enable_dsp(name)

    with sd.RawInputStream(device=idx,
                           samplerate=FS,
                           channels=CHANNELS,
                           dtype="int16") as stream:
        raw = stream.read(int(DURATION * FS))[0]

    # raw comes back as bytes ⇒ convert to int16 array
    samples = np.frombuffer(raw, dtype=np.int16)
    mono = samples[0::CHANNELS]          # keep channel 0

    sf.write(OUTFILE, mono, FS)
    print(f"Saved {OUTFILE}")


if __name__ == "__main__":
    main()
