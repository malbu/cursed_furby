#!/usr/bin/env python3
import re, time, serial, sys, itertools

PORT        = '/dev/ttyACM0'   
BAUD        = 115200
LOCK_COUNT  = 3                # need 3 good frames to declare lock
UNLOCK_COUNT= 5                # 5 bad frames to drop lock
DIST_LIMIT  = 150.0            # presence‑only
PRINT_EVERY = 1.0              # seconds between prints while locked
delta_HEART     = 3.0              # bpm change needed to reprint sooner
delta_BREATH    = 1.0              # bpm change needed to reprint sooner


RX = re.compile(
    r"Respiratory rate':.*?state ([\d\.]+)"
    r".*?heart rate':.*?state ([\d\.]+)"
    r".*?Distance .*?state ([\d\.]+) cm"
)

def parse(line:str):
    m = RX.search(line)
    return tuple(map(float, m.groups())) if m else None

ser = serial.Serial(PORT, BAUD, timeout=0.2)
print(f"[INFO] Listening on {PORT} …")

good, bad     = 0, 0
locked        = False
last_print    = 0.0
last_heart    = last_breath = None

for raw in itertools.repeat(None):  
    line = ser.readline().decode('utf-8', 'ignore')
    data = parse(line)
    now  = time.time()
    if not data:
        continue                    # not a sensor frame

    breath, heart, dist = data
    good_frame = heart and breath and dist <= DIST_LIMIT

    # ---- lock / unlock bookkeeping ----
    if good_frame:
        good, bad = good + 1, 0
    else:
        good, bad = 0, bad + 1

    if not locked and good >= LOCK_COUNT:
        locked, last_print = True, 0           # force immediate print
        print("[LOCK] torso detected")
    elif locked and bad >= UNLOCK_COUNT:
        locked = False
        print("[UNLOCK] target lost")

    
    if locked:
        heart_change  = last_heart  is None or abs(heart  - last_heart)  > delta_HEART
        breath_change = last_breath is None or abs(breath - last_breath) > delta_BREATH
        time_ok       = now - last_print >= PRINT_EVERY
        if (heart_change or breath_change or time_ok):
            print(f'{{"t":{now:.0f},"heart":{heart:.1f},"breath":{breath:.1f},"dist":{dist:.1f}}}')
            last_print, last_heart, last_breath = now, heart, breath
