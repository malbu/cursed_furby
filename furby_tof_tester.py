import time
import board
import busio
import adafruit_vl53l1x

"""Wiring:
SCL to GPIO3 (Pin 5)

SDA to GPIO2 (Pin 3)

VIN to 3.3V or 5V
GND to GND

"""


i2c = busio.I2C(board.SCL, board.SDA)


vl53 = adafruit_vl53l1x.VL53L1X(i2c)

# config
vl53.distance_mode = 2        # 1 = long (up to 4m), 2 = short (better accuracy at <1.3m)
vl53.timing_budget = 100      # milliseconds
vl53.start_ranging()


while True:
    distance_mm = vl53.distance
    print(f"Distance: {distance_mm / 10:.1f} cm")
    time.sleep(0.2)
