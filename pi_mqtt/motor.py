import pigpio, logging

log = logging.getLogger("motor")
IN1, IN2, ENA = 27, 22, 13  # BCM pins for L298N

class Motor:
    def __init__(self, duty_cycle: int = 100):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon not running")
        self.duty = int(255 * duty_cycle / 100)
        for p in (IN1, IN2, ENA):
            self.pi.set_mode(p, pigpio.OUTPUT)
        self.stop()

    def start(self):
        log.info("Motor start")
        self.pi.write(IN1, 1)
        self.pi.write(IN2, 0)
        self.pi.set_PWM_dutycycle(ENA, self.duty)

    def stop(self):
        log.info("Motor stop")
        self.pi.set_PWM_dutycycle(ENA, 0)
        self.pi.write(IN1, 0)
        self.pi.write(IN2, 0)

    def shutdown(self):
        self.stop()
        self.pi.stop()