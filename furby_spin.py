import RPi.GPIO as GPIO
from time import sleep
#using L298N driver

IN1, IN2, ENA = 27, 22, 13  
GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, ENA], GPIO.OUT)

pwm = GPIO.PWM(ENA, 5000)  
pwm.start(0)                

try:
    print("Forward")
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(80)  # ~80 % speed
    sleep(5)

    print("Reverse")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    sleep(5)

finally:
    pwm.stop(0)
    GPIO.cleanup()
