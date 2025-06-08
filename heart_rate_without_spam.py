#!/usr/bin/env python3
import re, time, serial, sys, itertools

PORT        = '/dev/ttyACM0'   
BAUD        = 115200

LOCK_COUNT  = 3
UNLOCK_COUNT= 5
DIST_LIMIT  = 150.0            # cm
PRINT_EVERY = 1.0              # s
delta_HEART     = 3.0              # bpm
delta_BREATH    = 1.0

# --- three regexes, one per metric -----------------------------------------
RX_BREATH  = re.compile(r"respiratory rate':.+?state ([\d\.]+)", re.I)
RX_HEART   = re.compile(r"heart rate':.+?state ([\d\.]+)",        re.I)
RX_DIST    = re.compile(r"distance to detection object':.+?state ([\d\.]+)", re.I)

def extract(line:str):
    """Return (label, value) or None."""
    for lbl, rx in (('breath', RX_BREATH),
                    ('heart' , RX_HEART ),
                    ('dist'  , RX_DIST )):
        if m := rx.search(line):
            return lbl, float(m.group(1))
    return None

ser = serial.Serial(PORT, BAUD, timeout=0.3)
print(f"[INFO] Listening on {PORT} …")

buf           = {}             # holds the most recent breath/heart/dist
good = bad    = 0
locked        = False
last_print    = 0.0
last_heart    = last_breath = None

for _ in itertools.repeat(None):
    raw  = ser.readline()
    if not raw:
        continue

    decoded = raw.decode('utf-8', 'ignore')
    got     = extract(decoded)
    if not got:
        continue

    lbl, val = got
    buf[lbl] = val

    # Wait until we have all three keys at least once
    if len(buf) < 3:
        continue

    breath = buf['breath']
    heart  = buf['heart']
    dist   = buf['dist']
    buf    = {}                 # clear for next trio

    good_frame = heart and breath and dist <= DIST_LIMIT

    # ---- lock / unlock bookkeeping ----
    if good_frame:
        good, bad = good + 1, 0
    else:
        good, bad = 0, bad + 1

    if not locked and good >= LOCK_COUNT:
        locked, last_print = True, 0
        print("[LOCK] torso detected")
    elif locked and bad >= UNLOCK_COUNT:
        locked = False
        print("[UNLOCK] target lost")

    # ---- throttled reporting while locked ----
    now = time.time()
    if locked:
        heart_change  = last_heart  is None or abs(heart  - last_heart)  > delta_HEART
        breath_change = last_breath is None or abs(breath - last_breath) > delta_BREATH
        time_ok       = now - last_print >= PRINT_EVERY
        if heart_change or breath_change or time_ok:
            print(f'{{"t":{now:.0f},"heart":{heart:.1f},'
                  f'"breath":{breath:.1f},"dist":{dist:.1f}}}')
            last_print, last_heart, last_breath = now, heart, breath
