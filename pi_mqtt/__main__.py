import asyncio, logging
from .sensor_mr60bha2 import SensorTask
from .mqtt_client import MQTTClient

logging.basicConfig(level=logging.INFO)

async def main():
    q = asyncio.Queue()
    sensor = SensorTask(q)
    mqttc  = MQTTClient(q)
    await asyncio.gather(sensor.run(), mqttc.pump())

if __name__ == "__main__":
    asyncio.run(main())