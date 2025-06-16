import time
import serial

# Initialize MR60BHA2 on USB Serial
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  
print("MR60BHA2 initialized. Listening for vital signs...\n")

while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(f"[MR60BHA2] {line}")
    time.sleep(0.1)
