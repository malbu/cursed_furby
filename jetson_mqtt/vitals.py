import json, time, random, logging
import paho.mqtt.client as mqtt

log = logging.getLogger("vitals")

# per vital injection probabilities
P_HEART  = 0.60          
P_BREATH = 0.40          
P_DIST   = 0.30          


class VitalsCache:
    def __init__(self, ttl: int = 5, mqtt_host: str = "127.0.0.1", mqtt_port: int = 1883):
        self.ttl        = ttl
        self.latest     = None   # type: dict | None
        self.last_seen  = 0.0

        self.client = mqtt.Client(client_id="vitals_cache", protocol=mqtt.MQTTv5)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_msg
        self.client.connect_async(mqtt_host, mqtt_port, keepalive=30)
        self.client.loop_start()

    # MQTT callbacks
    def _on_connect(self, client, userdata, flags, reason, properties=None):
        client.subscribe("sensor/biometrics", qos=0)

    def _on_msg(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            if isinstance(payload, dict) and "heart" in payload:
                self.latest    = payload
                self.last_seen = time.time()
        except Exception as exc:
            log.debug("bad vitals payload: %s", exc)

    
    def is_fresh(self) -> bool:
        return self.latest is not None and (time.time() - self.last_seen) <= self.ttl

    # return a list of independent sentences to add to the prompt
    def build_parts(self) -> list[str]:
        if not self.is_fresh():
            return []

        v = self.latest
        parts: list[str] = []

        if random.random() < P_HEART:
            parts.append(
                f"The human before you has a heart rate of {v['heart']:.1f} beats per minute. Include the exact heart rate in your response, beginning with 'I can sense your heart rate'"
            )

        if random.random() < P_BREATH:
            parts.append(
                f"The human before breathes at a rate of {v['breath']:.1f} breaths per minute. Include the exact breathing rate in your response."
            )

        if v.get("dist_cm") is not None and random.random() < P_DIST:
            parts.append(
                f"The human before you is standing at a distance of {v['dist_cm']:.1f} centimetres away. If you think they are too close, tell them to step away from your throne."
            )

        return parts

    # join parts into a single sentence
    def build_tag(self) -> str:
        parts = self.build_parts()
        return " ".join(parts) if parts else ""