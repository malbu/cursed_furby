import re, time, serial, json, asyncio, logging
from datetime import datetime, timezone

log = logging.getLogger("sensor")

# match same verbose strings as test script
RX_HEART  = re.compile(r"heart rate'.*?state\s+([\d\.]+)", re.I)
RX_BREATH = re.compile(r"respiratory rate'.*?state\s+([\d\.]+)", re.I)
RX_DIST   = re.compile(r"distance to detection object'.*?state\s+([\d\.]+)", re.I)

PRINT_EVERY  = 0.5   # seconds
UNLOCK_DELAY = 3.0   # seconds with heart==0 before unlock
DELTA_HEART  = 3.0   # bpm
DELTA_BR     = 1.0   # rpm

PORT, BAUD = "/dev/ttyACM0", 115200

# helpers
iso8601 = lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")

def mov_avg(prev, cur):
    return cur if prev is None else (prev + cur) / 2.0

class SensorTask:
    """Reads MR60BHA2 serial lines and puts JSON payloads into an asyncio.Queue"""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.ser = serial.Serial(PORT, BAUD, timeout=0.3)
        log.info("Listening on %s", PORT)
        # rolling state
        self.last_print = 0.0
        self.lock_time  = 0.0
        self.heart_avg = self.breath_avg = self.dist_avg = None

    async def run(self):
        while True:
            raw = self.ser.readline()
            if not raw:
                await asyncio.sleep(0.01)
                continue
            txt = raw.decode("utf-8", "ignore")
            got = False
            now = time.time()

            # Parse metrics
            if m := RX_HEART.search(txt):
                heart = float(m.group(1))
                got = True
                if heart > 0:
                    self.lock_time = now
                self.heart_avg = mov_avg(self.heart_avg, heart)

            if m := RX_BREATH.search(txt):
                breath = float(m.group(1))
                got = True
                self.breath_avg = mov_avg(self.breath_avg, breath)

            if m := RX_DIST.search(txt):
                dist = float(m.group(1))
                got = True
                self.dist_avg = mov_avg(self.dist_avg, dist)

            if not got:
                continue  # line had no useful fields

            locked = (now - self.lock_time) < UNLOCK_DELAY

            if locked and self.heart_avg and self.breath_avg:
                changed = (
                    abs(self.heart_avg - heart)  > DELTA_HEART or
                    abs(self.breath_avg - breath) > DELTA_BR
                )
                time_ok = (now - self.last_print) >= PRINT_EVERY
                if changed or time_ok:
                    payload = {
                        "ts":      iso8601(now),
                        "heart":   round(self.heart_avg, 1),
                        "breath":  round(self.breath_avg, 1),
                    }
                    if self.dist_avg is not None:
                        payload["dist_cm"] = round(self.dist_avg, 1)
                    await self.queue.put(json.dumps(payload))
                    self.last_print = now

            elif not locked and (now - self.last_print) > UNLOCK_DELAY:
                await self.queue.put("[UNLOCK] target lost")
                self.last_print = now