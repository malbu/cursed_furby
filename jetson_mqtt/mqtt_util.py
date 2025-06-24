import logging, asyncio
import paho.mqtt.client as mqtt
from contextlib import contextmanager

log = logging.getLogger("mqtt_util")

class MQTTOfflineError(RuntimeError):
    pass

class MQTTWrapper:
    def __init__(self, host="127.0.0.1", port=1883):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="jetson_helper")
        self.client.reconnect_delay_set(2, 30)
        self._connected = asyncio.Event()
        self.client.on_connect = lambda c,u,f,rc,p=None: self._connected.set()
        self.client.on_disconnect = lambda c,u,rc,p=None: self._connected.clear()
        self.client.connect_async(host, port, keepalive=30)
        self.client.loop_start()

    def publish(self, topic, payload, qos=0):
        if not self._connected.is_set():
            raise MQTTOfflineError
        self.client.publish(topic, payload, qos=qos)

    def subscribe(self, topic, cb):
        def handler(client, userdata, msg):
            cb(msg)
        self.client.message_callback_add(topic, handler)
        self.client.subscribe(topic)

_mqtt_singleton = None

def get_mqtt(host="127.0.0.1", port=1883):
    global _mqtt_singleton
    if _mqtt_singleton is None:
        _mqtt_singleton = MQTTWrapper(host, port)
    return _mqtt_singleton

@contextmanager
def motor_context(topic="furby/motor/cmd"):
    mqtt = get_mqtt()
    try:
        mqtt.publish(topic, "start")
    except MQTTOfflineError:
        log.warning("Pi offline – motor start skipped")
    try:
        yield
    finally:
        try:
            mqtt.publish(topic, "stop")
        except MQTTOfflineError:
            log.warning("Pi offline – motor stop skipped")