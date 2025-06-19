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
vl53.timing_budget = 50      # milliseconds
vl53.start_ranging()


while True:
    if vl53.data_ready:
        distance_cm = vl53.distance  # returns None when out of range
        if distance_cm is not None:
            print(f"Distance: {distance_cm:.1f} cm")
        else:
            print("Distance: --- (out of range)")
        vl53.clear_interrupt()
    time.sleep(0.1)
