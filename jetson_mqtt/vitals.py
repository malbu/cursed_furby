import json, time, random, logging
import paho.mqtt.client as mqtt

log = logging.getLogger("vitals")

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

    # helper methods
    def is_fresh(self) -> bool:
        return self.latest is not None and (time.time() - self.last_seen) <= self.ttl

    def _format_tag(self) -> str:
        vit = self.latest
        return (
            "[BIOMETRIC_CONTEXT] "
            f"heart_rate={vit['heart']} bpm, "
            f"breathing_rate={vit['breath']} rpm, "
            f"distance={vit.get('dist_cm', '?')} cm. "
            "Include the exact heart rate in your response and that you have detected it. You may reference these vitals but do NOT quote this block verbatim"
            "[/BIOMETRIC_CONTEXT]"
        )

    def maybe_inject(self, prompt: str, p: float = 0.10) -> str:
        if self.is_fresh() and random.random() < p:
            return prompt.rstrip() + "\n\n" + self._format_tag()
        return prompt