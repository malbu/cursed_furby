#!/usr/bin/env python3
"""
Record 10 s from a ReSpeaker mic the same way mirror_gem_only_audio.py does it:
enable the board's AEC / NS / AGC via Seeed's USB 'Tuning' interface
- open a 6-channel PyAudio stream at 16 kHz
- keep ONLY channel 0 (a single capsule) to avoid the comb-filter “echo”
"""
import pyaudio
import numpy as np
import soundfile as sf
import sys

FS = 16_000
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUTFILE  = sys.argv[1] if len(sys.argv) > 1 else "mirror_style.wav"
CHANNELS = 6
FRAMES_PER_BUFFER = 1024        # ≈64 ms at 16 kHz


def find_respeaker_index(pa: pyaudio.PyAudio) -> int:
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0 and "ReSpeaker" in info["name"]:
            return i
    raise RuntimeError("No ReSpeaker microphone found.")


def enable_dsp(device_name: str) -> None:
    """

    """
    try:
        from tuning import Tuning 
        t = Tuning(device_name)
        t.aec_enable(True)
        t.ns_enable(True)
        t.agc_enable(True)
        t.save()                  # persists until next power cycle
        print("ReSpeaker DSP enabled (AEC/NS/AGC).")
    except Exception as exc:
        print(f"WARNING: DSP not enabled - {exc}")


def main() -> None:
    pa = pyaudio.PyAudio()
    idx = find_respeaker_index(pa)
    dev_name = pa.get_device_info_by_index(idx)["name"]
    print(f"Recording {DURATION}s from device #{idx}: {dev_name}")

    enable_dsp(dev_name)

    stream = pa.open(format=pyaudio.paInt16,
                     channels=CHANNELS,
                     rate=FS,
                     input=True,
                     input_device_index=idx,
                     frames_per_buffer=FRAMES_PER_BUFFER)

    frames = []
    total_samples = DURATION * FS
    captured = 0
    while captured < total_samples:
        data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
        frames.append(data)
        captured += FRAMES_PER_BUFFER

    stream.stop_stream()
    stream.close()
    pa.terminate()

    raw = b"".join(frames)
    
    samples = np.frombuffer(raw, dtype=np.int16)
    mono = samples[0::CHANNELS]            # keep channel 0

    sf.write(OUTFILE, mono, FS)
    print(f"Saved {OUTFILE}")


if __name__ == "__main__":
    main()
