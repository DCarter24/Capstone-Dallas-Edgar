import math
import time
import busio
from board import SCL, SDA
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from adafruit_rplidar import RPLidar

class LidarObjectDetection:
    NEUTRAL_ANGLE = 90
    SLOW_SPEED_PERCENT = 0.14
    TURN_SPEED_PERCENT = 0.20
    STOP_SPEED = 0
    DETECTION_RANGE_MIN = 0.1  # meters
    DETECTION_RANGE_MAX = 2.0  # meters
    RIGHT_TURN_ANGLE = 135
    LEFT_TURN_ANGLE = 45

    def __init__(self, port='/dev/ttyUSB0'):
        # Initialize PCA9685 and servo
        self.pca = self.init_servo_motor()
        self.steering_servo = servo.Servo(self.pca.channels[14])
        self.steering_servo.angle = self.NEUTRAL_ANGLE

        # Initialize RPLidar
        self.lidar = RPLidar(None, port, timeout=3)
        print("Lidar initialized.")

    def init_servo_motor(self):
        i2c_bus = busio.I2C(SCL, SDA)
        pca = PCA9685(i2c_bus)
        pca.frequency = 100
        return pca

    def motor_speed(self, percent):
        # Convert a percentage to a PCA9685 duty cycle
        # Adjusting the formula as needed for your motor controller
        speed = (percent * 3277) + (65535 * 0.15)
        self.pca.channels[15].duty_cycle = math.floor(speed)
        print(f"Motor Speed: {percent:.2f}")

    def run(self):
        try:
            for scan in self.lidar.iter_scans():
                for (_, angle, distance_mm) in scan:
                    # Convert distance to meters
                    distance_m = distance_mm / 1000.0
                    # Check if distance is within the detection range
                    if self.DETECTION_RANGE_MIN < distance_m < self.DETECTION_RANGE_MAX:
                        # Focus on front-right quadrant: (270-360°) and (0-90°)
                        if 270 <= angle <= 360 or 0 <= angle <= 90:
                            print(f"Object detected at angle {angle:.1f}° and distance {distance_m:.2f}m")
                            self.decide_maneuver(angle, distance_m)
        except KeyboardInterrupt:
            print("Stopping due to KeyboardInterrupt.")
        finally:
            self.lidar.stop()
            self.lidar.disconnect()
            print("Lidar stopped and disconnected.")

    def decide_maneuver(self, angle, distance):
        # Determine which action to take based on angle and distance
        if (0 < angle <= 45) or (angle >= 315):
            self.handle_ahead_object(angle, distance)
        elif 45 < angle <= 90:
            self.turn_left()
        elif 270 < angle < 315:
            self.turn_right()

    def handle_ahead_object(self, angle, distance):
        # If the object is very close, stop; otherwise slow forward
        if distance < 0.02:
            self.stop()
        else:
            self.move_forward_slow()

    def move_forward(self):
        self.motor_speed(self.SLOW_SPEED_PERCENT)
        self.steering_servo.angle = self.NEUTRAL_ANGLE

    def move_forward_slow(self):
        self.motor_speed(self.TURN_SPEED_PERCENT)
        self.steering_servo.angle = self.NEUTRAL_ANGLE

    def turn_left(self):
        self.steering_servo.angle = self.LEFT_TURN_ANGLE
        self.motor_speed(self.TURN_SPEED_PERCENT)

    def turn_right(self):
        self.steering_servo.angle = self.RIGHT_TURN_ANGLE
        self.motor_speed(self.TURN_SPEED_PERCENT)

    def stop(self):
        self.motor_speed(self.STOP_SPEED)


def main():
    detector = LidarObjectDetection(port='/dev/ttyUSB0')
    detector.run()
  


if __name__ == '__main__':
    main()
