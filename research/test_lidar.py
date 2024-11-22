#!/usr/bin/env python3

import math
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
import time
from sensor_msgs.msg import LaserScan
import rclpy
from rclpy.node import Node

# Motor Initialization
def Servo_Motor_Initialization():
    """
    Initializes the I2C bus and the PCA9685 motor controller.
    """
    i2c_bus = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c_bus)
    pca.frequency = 100
    return pca

def Motor_Start(pca):
    """
    Initialize motor settings as per the device instructions.
    """
    input("Press and hold EZ button. Once the LED turns red, release the button and press ENTER.")
    Motor_Speed(pca, 1)
    time.sleep(2)
    input("If the LED blinked TWICE, press ENTER.")
    Motor_Speed(pca, -1)
    time.sleep(2)
    input("Now the LED should be solid green. Press ENTER to proceed.")

def Motor_Speed(pca, percent):
    """
    Converts a percentage value (-1 to 1) to a 16-bit duty cycle
    and sets the motor speed accordingly.
    """
    speed = ((percent) * 3277) + 65535 * 0.15
    pca.channels[15].duty_cycle = math.floor(speed)
    print(f"Motor speed set to {speed/65535:.2f} (16-bit duty cycle)")

# LIDAR Node
class LidarObjectDetection(Node):
    def __init__(self):
        super().__init__('lidar_object_detection')
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10
        )
        self.pca = Servo_Motor_Initialization()
        Motor_Start(self.pca)
        self.neutral_speed = 0
        self.forward_speed = 0.15
        self.reverse_speed = -0.15
        self.turn_speed = 0.1  # Turning speed

    def lidar_callback(self, msg):
        """
        Callback to process LIDAR data and control motor accordingly.
        """
        detected_objects = self.process_lidar_data(msg)
        if detected_objects:
            self.decide_maneuver(detected_objects)
        else:
            self.move_forward()

    def process_lidar_data(self, scan_data):
        """
        Process LIDAR data to detect objects in the front range.
        """
        detected_objects = []
        for i, distance in enumerate(scan_data.ranges):
            if 0.2 < distance < 2.0:  # Filter distance range
                angle = math.degrees(scan_data.angle_min + i * scan_data.angle_increment) % 360
                if angle >= 270 or angle <= 90:  # Filter front range
                    detected_objects.append((angle, distance))
        return detected_objects

    def decide_maneuver(self, detected_objects):
        """
        Decide the car's movement based on object positions.
        """
        closest_object = min(detected_objects, key=lambda x: x[1])  # Closest object
        angle, distance = closest_object

        print(f"Object detected at {angle:.2f}° and {distance:.2f} meters.")

        if angle < 45 or (angle > 315):  # Object is slightly left
            print("Turning right...")
            Motor_Speed(self.pca, self.turn_speed)  # Turn right
            time.sleep(1)
        elif 45 <= angle <= 135:  # Object is directly in front
            print("Reversing...")
            Motor_Speed(self.pca, self.reverse_speed)  # Reverse
            time.sleep(2)
        elif angle > 225:  # Object is slightly right
            print("Turning left...")
            Motor_Speed(self.pca, -self.turn_speed)  # Turn left
            time.sleep(1)
        else:
            print("Moving forward...")
            self.move_forward()

    def move_forward(self):
        """
        Move the car forward.
        """
        print("Car moving forward...")
        Motor_Speed(self.pca, self.forward_speed)
        time.sleep(3)
        Motor_Speed(self.pca, self.neutral_speed)

# Main
def main(args=None):
    rclpy.init(args=args)
    lidar_object_detection = LidarObjectDetection()
    rclpy.spin(lidar_object_detection)
    lidar_object_detection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
