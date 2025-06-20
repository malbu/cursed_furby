import RPi.GPIO as GPIO
from time import sleep
# using L298N driver with home-position switch support

# Motor control pins (BCM numbering)
IN1, IN2, ENA = 27, 22, 13


SWITCH_PIN = 17 

# GPIO initialisation
GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, ENA], GPIO.OUT)
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # use internal pull-up resistor


pwm = GPIO.PWM(ENA, 5000)
pwm.start(0)


def motor_forward(duty_cycle: int = 100):
    """Spin motor forward at the given duty-cycle (0-100 %)."""
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(duty_cycle)


def motor_stop():
    """Stop motor - coast by default, brake if both lines LOW."""
    pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)


try:
    
    home_state = GPIO.input(SWITCH_PIN)  

    
    motor_forward()
    while GPIO.input(SWITCH_PIN) == home_state:
        sleep(0.01)
    motor_stop()
    sleep(0.1)  

   
    while GPIO.input(SWITCH_PIN) != home_state:
        
        motor_forward()
        sleep(1)
        
        motor_stop()
        for _ in range(int(2 / 0.01)):
            if GPIO.input(SWITCH_PIN) == home_state:
                break
            sleep(0.01)

    
    motor_stop()
    print("New home position reached. Motor stopped.")

finally:
    pwm.stop()
    GPIO.cleanup()
