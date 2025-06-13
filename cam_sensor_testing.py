#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time, sys


GEAR_PIN = 18          # TXS0102 A1 (B1 from Furby Pin 22)
HOME_PIN = 23          # TXS0102 A2 (B2 from Furby Pin 23)
AIN1     = 27          # DRV8833 IN1
AIN2     = 22          # DRV8833 IN2
PWMA     = 13          # DRV8833 PWMA  


GPIO.setmode(GPIO.BCM)
GPIO.setup([AIN1, AIN2, PWMA], GPIO.OUT)
GPIO.setup([GEAR_PIN, HOME_PIN], GPIO.IN, pull_up_down=GPIO.PUD_UP)

pwm = GPIO.PWM(PWMA, 5000)      # 5 kHz
pwm.start(0)

# edge counters
gear_cnt  = 0
home_cnt  = 0

def gear_cb(ch):
    global gear_cnt
    gear_cnt += 1

def home_cb(ch):
    global home_cnt
    home_cnt += 1

GPIO.add_event_detect(GEAR_PIN, GPIO.RISING,  callback=gear_cb,  bouncetime=1)
GPIO.add_event_detect(HOME_PIN, GPIO.FALLING, callback=home_cb, bouncetime=5)

def run_motor(fwd=True, seconds=5, duty=80):
    
    global gear_cnt, home_cnt
    gear_cnt = home_cnt = 0

    GPIO.output(AIN1, int(fwd))
    GPIO.output(AIN2, int(not fwd))
    pwm.ChangeDutyCycle(duty)

    start = time.time()
    while time.time() - start < seconds:
        print(f"\rTime {time.time()-start:4.1f}s | Gear {gear_cnt:4d} | Home {home_cnt}", end="")
        time.sleep(1)
    print()                        # newline
    pwm.ChangeDutyCycle(0)

try:
    print("== Forward 5 s ==")
    run_motor(fwd=True,  seconds=5, duty=80)

    time.sleep(0.5)
    print("== Reverse 5 s ==")
    run_motor(fwd=False, seconds=5, duty=80)

finally:
    pwm.stop()
    GPIO.cleanup()
    print("GPIO cleaned up - test complete.")

