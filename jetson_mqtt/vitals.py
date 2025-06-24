import json, time, random, logging
from dataclasses import dataclass
from .mqtt_util import get_mqtt

log = logging.getLogger("vitals")
_now = time.time

@dataclass
class Vitals:
    heart: float
    breath: float
    dist_cm: float | None
    ts: str

class VitalsCache:
    def __init__(self, ttl=5):
        self.ttl = ttl
        self.latest: Vitals | None = None
        self._last_seen = 0.0
        get_mqtt().subscribe("sensor/biometrics", self._on_msg)

    def _on_msg(self, msg):
        try:
            data = json.loads(msg.payload)
            self.latest = Vitals(**data)
            self._last_seen = _now()
        except Exception:
            # likely the "[UNLOCK]" string – ignore
            return

    def maybe_inject(self, prompt: str, p: float = 0.10) -> str:
        if self.latest and (_now() - self._last_seen) < self.ttl and random.random() < p:
            v = self.latest
            dist_str = f" {v.dist_cm:.1f}cm" if v.dist_cm is not None else ""
            return prompt + f"\n\n[HR:{v.heart} RR:{v.breath}{dist_str}]"
        return prompt