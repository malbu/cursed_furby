#!/usr/bin/env python3
import RPi.GPIO as GPIO, time


DIR1  = 27        # L298N IN1
DIR2  = 22        # L298N IN2
ENA   = 13        # L298N ENA  (PWM)

GEAR  = 18        # TXS0102 A1 <- Furby Pin 22
HOME  = 23        # TXS0102 A2 <- Furby Pin 23

GPIO.setmode(GPIO.BCM)
GPIO.setup([DIR1, DIR2, ENA], GPIO.OUT)
GPIO.setup([GEAR, HOME], GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)   
# GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)

pwm = GPIO.PWM(ENA, 5000)      # 5 kHz
pwm.start(0)

gear_cnt = 0
home_cnt = 0

def gear_cb(ch):
    global gear_cnt
    gear_cnt += 1

def home_cb(ch):
    global home_cnt
    home_cnt += 1

GPIO.add_event_detect(GEAR, GPIO.FALLING, callback=gear_cb,  bouncetime=5)
GPIO.add_event_detect(HOME, GPIO.RISING,  callback=home_cb, bouncetime=5)


def run(direction_high, seconds=5, duty=100):
    global gear_cnt, home_cnt
    gear_cnt = home_cnt = 0

    GPIO.output(DIR1, direction_high)
    GPIO.output(DIR2, not direction_high)
    pwm.ChangeDutyCycle(duty)

    start = time.time()
    while time.time() - start < seconds:
        print(f"\r{('FWD' if direction_high else 'REV')} t={time.time()-start:4.1f}s "
              f"| Gear {gear_cnt:4d} | Home {home_cnt}", end="")
        time.sleep(1)
    pwm.ChangeDutyCycle(0)
    print()

try:
    print("== Forward 5 s ==")
    run(True)

    time.sleep(0.5)
    print("== Reverse 5 s ==")
    run(False)

finally:
    pwm.stop()     # stop while the chip handle is still valid
    del pwm        # force destructor to run now
    GPIO.cleanup()
    print("Test finished, GPIO released.")
