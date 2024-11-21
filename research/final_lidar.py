#!/usr/bin/env python3

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math
import time
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685


class LidarObjectDetection(Node):
    def __init__(self):
        super().__init__('lidar_object_detection')

        # Creating a publisher to send Twist messages to the /cmd_vel topic
        self.steering_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Initialize PCA9685 and servo motor for speed control
        self.pca = self.servo_motor_initialization()

        # Debugging statement
        self.get_logger().info("Lidar Object Detection node initialized.")

        # Create a subscription to LIDAR data from the /scan topic
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',  # LIDAR topic
            self.lidar_callback,
            10
        )

    def servo_motor_initialization(self):
        """Initialize motor control"""
        i2c_bus = busio.I2C(SCL, SDA)
        pca = PCA9685(i2c_bus)
        pca.frequency = 100
        return pca

    def motor_speed(self, percent):
        """Control motor speed"""
        speed = ((percent) * 3277) + 65535 * 0.15
        self.pca.channels[15].duty_cycle = math.floor(speed)
        self.get_logger().info(f"Motor speed set to {percent*100:.1f}% (duty cycle: {speed / 65535:.2f})")

    def lidar_callback(self, msg):
        # Process the LIDAR data to detect objects
        detected_objects = self.process_lidar_data(msg)
        self.print_detected_objects(detected_objects)

        # If any objects are detected, decide on a maneuver
        if detected_objects:
            self.decide_maneuver(detected_objects)

    def process_lidar_data(self, msg):
        """Process LIDAR scan data and identify objects within range"""
        detected_objects = []
        for i, range_val in enumerate(msg.ranges[:360]):  # Ensure we only process up to 360 degrees
            if 0.2 < range_val < 1.5:  # Detect objects within this range
                angle = i  # Angle in degrees
                detected_objects.append((angle, range_val))  # Store angle and distance
        return detected_objects

    def decide_maneuver(self, detected_objects):
        """Decide speed and steering based on detected objects"""
        closest_object = min(detected_objects, key=lambda x: x[1])  # Closest object by distance
        angle = closest_object[0]
        distance = closest_object[1]

        # Create a Twist message to send steering commands
        cmd = Twist()
        self.get_logger().info(f"Closest object at angle: {angle}°, distance: {distance:.2f}m")

        # Adjust speed and turn rate based on object position and distance
        if distance < 0.5:  # Object too close
            cmd.linear.x = 0.0  # Stop
            cmd.angular.z = -1.0 if angle > 180 else 1.0  # Turn away from object
        elif 0.5 <= distance < 1.0:  # Object detected, but at a moderate distance
            cmd.linear.x = 0.5  # Slow forward
            cmd.angular.z = -0.5 if angle > 180 else 0.5  # Turn left/right
        else:  # No immediate obstacle
            cmd.linear.x = 1.0  # Full forward
            cmd.angular.z = 0.0  # Straight

        # Publish the steering command (Twist message)
        self.steering_publisher.publish(cmd)

    def print_detected_objects(self, detected_objects):
        """Prints all detected objects with angles and distances"""
        if detected_objects:
            print("\nDetected Objects:")
            print(f"{'Angle (degrees)':<20} {'Distance (meters)':<20}")
            print("-" * 40)
            for obj in detected_objects:
                angle_deg = obj[0]
                distance = obj[1]
                print(f"{angle_deg:<20} {distance:.2f}")
        else:
            print("No objects detected.\n")


def main(args=None):
    rclpy.init(args=args)
    lidar_object_detection = LidarObjectDetection()
    rclpy.spin(lidar_object_detection)
    lidar_object_detection.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
