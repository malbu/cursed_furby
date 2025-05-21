import time
import board
import busio
import serial
import adafruit_vl53l4cx

# initialize VL53L4CX on I2C 
i2c = busio.I2C(board.SCL, board.SDA)
vl53 = adafruit_vl53l4cx.VL53L4CX(i2c)
vl53.start_ranging()

# initialize MR60BHA2 on USB Serial
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

print("System initialized. Monitoring started...\n")

while True:
    # read distance from ToF sensor
    vl53.update()
    distance_mm = vl53.distance
    distance_m = distance_mm / 1000.0

    # report distance always
    print(f"[ToF] Distance: {distance_m:.2f} m")

    # only process mmWave if someone is close enough...
    if 0.3 < distance_m < 1.5:
        print("Target within vital zone. Reading MR60BHA2 data...")
        start_time = time.time()
        timeout = 2  # seconds

        while time.time() - start_time < timeout:
            if ser.in_waiting > 0:
                line = ser.readline().decode(errors='ignore').strip()
                if "heart_rate" in line or "breath_rate" in line:
                    print(f"[MR60BHA2] {line}")
    else:
        print("No one in vital zone. Skipping HR/BR check.")

    print("-" * 40)
    time.sleep(0.5)
