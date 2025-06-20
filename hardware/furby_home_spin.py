"""furby_home_spin.py – diagnostic tool for finding the motor‐on times
needed to move ONLY the mouth or ONLY the eyes.

How it works
-------------
1. Homes the cam using the limit switch.
2. Measures the time required for **one full revolution** at the chosen
   duty-cycle.
3. Divides that revolution into equal-time bursts (`STEP_MS`).
4. Executes each burst, stopping in-between so the user can observe which
   mechanisms moved.  After every burst it prints:

       • step index
       • cumulative elapsed time (ms)
       • approximate cam angle (°)

Adjust `STEP_MS`, `PAUSE_MS`, or `DUTY_CYCLE` below to taste.  Press
Ctrl-C at any time to abort safely.
"""

import time
from time import sleep

import RPi.GPIO as GPIO

# ---------------------------  USER SETTINGS  --------------------------- #
DUTY_CYCLE = 60   # % PWM duty while the motor is ON
STEP_MS     = 50   # motor-ON burst length (milliseconds)
PAUSE_MS    = 800  # observation pause after each burst (milliseconds)
PWM_FREQ    = 5000 # Hz on ENA pin
# ---------------------------------------------------------------------- #

# Motor control pins (BCM numbering)
IN1, IN2, ENA = 27, 22, 13

# Home/limit switch pin (pull-up fitted)
SWITCH_PIN = 17

# ---------------------------  GPIO SETUP  ------------------------------ #
GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, ENA], GPIO.OUT)
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

pwm = GPIO.PWM(ENA, PWM_FREQ)
pwm.start(0)

# ----------------------------  HELPERS  ------------------------------- #

def motor_forward(duty: int = DUTY_CYCLE) -> None:
    """Drive the cam forward at *duty* percent PWM."""
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(duty)


def motor_stop() -> None:
    """Cut PWM and de-energise IN1/IN2 (coast)."""
    pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)


def seek_home(timeout: float = 5.0) -> None:
    """Rotate until the switch toggles back to its *pressed* state."""
    start = time.time()
    initial_state = GPIO.input(SWITCH_PIN)

    # Leave the home position first, so we can detect it again later.
    motor_forward()
    while GPIO.input(SWITCH_PIN) == initial_state:
        if time.time() - start > timeout:
            raise RuntimeError("Timeout while leaving home position")
        sleep(0.005)

    # Now wait until we re-enter the original state = true home.
    while GPIO.input(SWITCH_PIN) != initial_state:
        if time.time() - start > timeout:
            raise RuntimeError("Timeout while seeking home position")
        sleep(0.005)

    motor_stop()
    print("Home located.")


def full_rotation_time() -> float:
    """Return the time (seconds) for one complete revolution."""
    print("Measuring time for one full rotation…")
    initial_state = GPIO.input(SWITCH_PIN)

    # Ensure we start exactly ON the switch.
    if initial_state != GPIO.LOW:
        seek_home()

    # Leave home and start timing.
    motor_forward()
    while GPIO.input(SWITCH_PIN) == initial_state:
        sleep(0.001)

    t0 = time.time()

    # Wait until we hit home again.
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


# ----------------------------  MAIN FLOW  ------------------------------ #

try:
    print("Starting diagnostic stepping routine…  Press Ctrl-C to abort.")

    # 1) Home the cam to a known zero.
    seek_home()

    # 2) Measure the base speed.
    rev_time = full_rotation_time()
    deg_per_ms = 360 / (rev_time * 1000)

    # 3) Begin stepping.
    step_s  = STEP_MS  / 1000.0
    pause_s = PAUSE_MS / 1000.0

    total_ms = 0.0
    step_idx = 0

    print("\n--- Beginning incremental rotation ---\n")
    while total_ms < rev_time * 1000:
        step_idx += 1

        rotate_for(step_s)
        total_ms += STEP_MS

        approx_deg = total_ms * deg_per_ms
        print(f"Step {step_idx:02d}: t = {total_ms:5.0f} ms   ≈ {approx_deg:6.1f} °")

        sleep(pause_s)

    print("\nDone. One full revolution (or slightly more) completed.")

except KeyboardInterrupt:
    print("\nUser abort (Ctrl-C).")

finally:
    motor_stop()
    pwm.stop()
    GPIO.cleanup()
    print("GPIO cleaned up. Bye!")
