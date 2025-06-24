import asyncio, logging
from .sensor_mr60bha2 import SensorTask
from .mqtt_client import MQTTClient
from .config import load as load_cfg

logging.basicConfig(level=logging.INFO)

async def main():
    cfg     = load_cfg().get("mqtt", {})
    broker  = cfg.get("broker_host", "mqtt.local")
    port    = cfg.get("broker_port", 1883)

    q       = asyncio.Queue()
    sensor  = SensorTask(q)
    mqttc   = MQTTClient(q, broker=broker, port=port)
    await asyncio.gather(sensor.run(), mqttc.pump())

if __name__ == "__main__":
    asyncio.run(main())