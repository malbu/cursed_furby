#!/usr/bin/env python3

import re, time, serial, itertools, argparse
from datetime import datetime, timezone

PORT, BAUD = "/dev/ttyACM0", 115200

PRINT_EVERY   = 1.0    # s between lines while locked
UNLOCK_DELAY  = 3.0    # s of heart=0 before "unlock"
delta_HEART, delta_BR = 3.0, 1.0
#unlock / confidence parameters 
DIST_UNLOCK_CM   = 160         # lean back threshold (don't unlock if torso still ≤160 cm)
FAR_CM           = 200         # consider far away/gone when distance > 200 cm
UNLOCK_FRAMES    = 3           # need this many consecutive far+no-HR frames
CONF_LOW_THRESH  = 100         # >100 cm -> confidence="low"

#regexes
RX_HEART  = re.compile(r"heart rate'.*?state\s+([\d\.]+)",          re.I)
RX_BREATH = re.compile(r"respiratory rate'.*?state\s+([\d\.]+)",    re.I)
RX_DIST   = re.compile(r"distance to detection object'.*?state\s+([\d\.]+)", re.I)

#helpers
def iso8601(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")

def mov_avg(prev: float | None, cur: float) -> float:
    return cur if prev is None else (prev + cur) / 2.0

#serial setup
ser = serial.Serial(PORT, BAUD, timeout=0.3)
print(f"[INFO] listening on {PORT}")

#state vars
last_print        = 0.0
last_dist_print   = 0.0
lock_time         = 0.0          
heart_avg    = breath_avg = dist_avg = None
unlock_counter    = 0          # debounce frames


parser = argparse.ArgumentParser(description="Biometric detection listener")
parser.add_argument("--raw", action="store_true", help="Output raw sensor values instead of averages")
args = parser.parse_args()

for _ in itertools.repeat(None):
    raw = ser.readline()
    if not raw:
        continue
    txt = raw.decode("utf-8", "ignore")

    got = False
    if m := RX_HEART.search(txt):
        heart = float(m.group(1))
        got   = True
        if heart > 0:
            lock_time = time.time()
        heart_avg = mov_avg(heart_avg, heart)

    if m := RX_BREATH.search(txt):
        breath     = float(m.group(1))
        got        = True
        breath_avg = mov_avg(breath_avg, breath)

    if m := RX_DIST.search(txt):
        dist      = float(m.group(1))
        got       = True
        dist_avg  = mov_avg(dist_avg, dist)

    if not got:                     # line had none of desired metrics
        continue

    now = time.time()

    # update lock timer on either non-zero HR 
    if (heart_avg is not None and heart_avg > 0):
        lock_time = now

    # primary lock flag: torso present AND within DIST_UNLOCK_CM
    locked = (dist_avg is None or dist_avg <= DIST_UNLOCK_CM)

    #  unlock debounce 
    far_enough = dist_avg is not None and dist_avg > FAR_CM
    no_heart   = heart_avg is not None and heart_avg == 0

    if far_enough and no_heart:
        unlock_counter += 1
    else:
        unlock_counter = 0

    if unlock_counter >= UNLOCK_FRAMES and locked:
        locked = False
        print("[UNLOCK] target lost")
        last_print = now

    if locked and heart_avg and breath_avg:
        # throttle output
        changed = (
            abs(heart_avg  - heart)  > delta_HEART or
            abs(breath_avg - breath) > delta_BR
        )
        time_ok = (now - last_print) >= PRINT_EVERY
        if changed or time_ok:
            payload = {
                "ts": iso8601(now),
                "heart":  round(heart_avg, 1),
                "breath": round(breath_avg, 1),
                "dist_cm": round(dist_avg, 1) if dist_avg is not None else None,
                "avg": not args.raw,
                "confidence": "low" if (dist_avg is not None and dist_avg > CONF_LOW_THRESH) else "high",
            }
            print(payload)
            last_print = now

    elif not locked and (now - last_print) > UNLOCK_DELAY:
        print("[UNLOCK] target lost")
        last_print = now
