import asyncio, logging
import paho.mqtt.client as mqtt
from .motor import Motor

log = logging.getLogger("mqtt_client")

class MQTTClient:
    def __init__(self, queue: asyncio.Queue, broker="mqtt.local", port=1883):
        self.queue = queue
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="pi_agent")
        self.client.reconnect_delay_set(2, 30)
        self.client.will_set("pi/status", "offline", retain=True)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_msg
        self.motor = Motor()
        self.client.connect_async(broker, port, keepalive=30)
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        log.info("Connected to broker rc=%s", rc)
        client.publish("pi/status", "online", retain=True)
        client.subscribe("furby/motor/cmd")

    def _on_msg(self, client, userdata, msg):
        if msg.topic == "furby/motor/cmd":
            cmd = msg.payload.decode()
            if cmd == "start":
                self.motor.start()
            elif cmd == "stop":
                self.motor.stop()

    async def pump(self):
        while True:
            pkt = await self.queue.get()
            self.client.publish("sensor/biometrics", pkt, qos=0)