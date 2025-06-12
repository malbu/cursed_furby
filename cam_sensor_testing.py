import RPi.GPIO as GPIO, time
GEAR = 18; HOME = 23; LED_EN = None  # LED always on 

GPIO.setmode(GPIO.BCM)
GPIO.setup([GEAR, HOME], GPIO.IN, pull_up_down=GPIO.PUD_UP)

def pulse(pin):
    print("Waiting for pulses")
    try:
        last = time.time()
        while True:
            if not GPIO.input(pin):  # 
                dt = time.time() - last
                last = time.time()
                print(f"Edge, Δt={dt*1000:.1f} ms")
                while not GPIO.input(pin): pass
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()

pulse(GEAR)
