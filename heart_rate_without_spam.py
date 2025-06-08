#!/usr/bin/env python3
import re, time, serial, itertools

PORT  = '/dev/ttyACM0'
BAUD  = 115200

PRINT_EVERY   = 1.0   # seconds while locked
UNLOCK_DELAY  = 3.0   # seconds of 0‑BPM before declaring unlock
delta_HEART, delta_BR = 3.0, 1.0  # change needed to reprint sooner

RX_HEART  = re.compile(r"heart rate'.*?state\s+([\d\.]+)",  re.I)
RX_BREATH = re.compile(r"respiratory rate'.*?state\s+([\d\.]+)", re.I)

ser = serial.Serial(PORT, BAUD, timeout=0.3)
print(f"[INFO] listening on {PORT}")

last_print = 0.0
last_heart = last_breath = None
lock_time  = 0.0        # last time saw a non zero heart

for _ in itertools.repeat(None):
    raw = ser.readline()
    if not raw:
        continue
    txt = raw.decode('utf‑8', 'ignore')

    h = RX_HEART.search(txt)
    b = RX_BREATH.search(txt)

    if not (h or b):
        continue                      # not a vital‑sign line

    now = time.time()

    if h:
        heart = float(h.group(1))
        if heart > 0:
            lock_time = now
        else:
            heart = 0.0
        last_heart = heart

    if b:
        breath = float(b.group(1))
        last_breath = breath

    locked = (now - lock_time) < UNLOCK_DELAY

    if locked:
        # throttle prints
        heart_change  = last_heart  is None or abs(last_heart  - heart ) > delta_HEART
        breath_change = last_breath is None or abs(last_breath - breath) > delta_BR
        time_ok       = now - last_print >= PRINT_EVERY
        if heart_change or breath_change or time_ok:
            print(f'{{"t":{now:.0f},"heart":{heart:.1f},"breath":{breath:.1f}}}')
            last_print = now
    else:
        if now - last_print > UNLOCK_DELAY:
            print("[UNLOCK] target lost")
            last_print = now
