"""diagnostic  for finding the motor on times
needed to move ONLY the mouth or ONLY the eyes
approach abandoned because of slippage; apporach still feasible if default motor replace with servo
"""

import time
from time import sleep

import RPi.GPIO as GPIO


DUTY_CYCLE = 60   
STEP_MS     = 50   
PAUSE_MS    = 800  
PWM_FREQ    = 5000  


# control pins
IN1, IN2, ENA = 27, 22, 13

# home/limit switch pin; pull
SWITCH_PIN = 17


GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, ENA], GPIO.OUT)
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

pwm = GPIO.PWM(ENA, PWM_FREQ)
pwm.start(0)



def motor_forward(duty: int = DUTY_CYCLE) -> None:
    
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(duty)


def motor_stop() -> None:
    
    pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)


def seek_home(timeout: float = 5.0) -> None:
    
    start = time.time()
    initial_state = GPIO.input(SWITCH_PIN)

    
    motor_forward()
    while GPIO.input(SWITCH_PIN) == initial_state:
        if time.time() - start > timeout:
            raise RuntimeError("Timeout while leaving home position")
        sleep(0.005)

    
    while GPIO.input(SWITCH_PIN) != initial_state:
        if time.time() - start > timeout:
            raise RuntimeError("Timeout while seeking home position")
        sleep(0.005)

    motor_stop()
    print("Home located.")


def full_rotation_time() -> float:
    
    print("Measuring time for one full rotatio")
    initial_state = GPIO.input(SWITCH_PIN)

    # start exactly ON the switch.
    if initial_state != GPIO.LOW:
        seek_home()

    # start timing
    motor_forward()
    while GPIO.input(SWITCH_PIN) == initial_state:
        sleep(0.001)

    t0 = time.time()

    # wait until home is hit again
    while GPIO.input(SWITCH_PIN) != initial_state:
        sleep(0.001)

    elapsed = time.time() - t0
    motor_stop()

    print(f"One revolution ≈ {elapsed:.3f} s")
    return elapsed


def rotate_for(sec: float) -> None:
    motor_forward()
    sleep(sec)
    motor_stop()




try:
    

    
    seek_home()

   
    rev_time = full_rotation_time()
    deg_per_ms = 360 / (rev_time * 1000)

    
    step_s  = STEP_MS  / 1000.0
    pause_s = PAUSE_MS / 1000.0

    total_ms = 0.0
    step_idx = 0

    print("\n--- beginning incremental rotation ---\n")
    while total_ms < rev_time * 1000:
        step_idx += 1

        rotate_for(step_s)
        total_ms += STEP_MS

        approx_deg = total_ms * deg_per_ms
        print(f"Step {step_idx:02d}: t = {total_ms:5.0f} ms   ≈ {approx_deg:6.1f} °")

        sleep(pause_s)

    print("\nDone. One full revolution complete")

except KeyboardInterrupt:
    print("\nuser abort")

finally:
    motor_stop()
    pwm.stop()
    GPIO.cleanup()
    
