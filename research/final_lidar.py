#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import time
import math
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
        self.get_logger().info("LidarObjectDetection node initialized.")
        
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
        self.get_logger().info(f"Motor speed set to {percent*100}% (duty cycle: {speed / 65535:.2f})")

    def lidar_callback(self, msg):
        # Process the LIDAR data to detect objects
        detected_objects = self.process_lidar_data(msg)
        self.print_detected_objects(detected_objects)
        
        # If any objects are detected, decide on a maneuver
        if detected_objects:
            self.decide_maneuver(detected_objects)

    def process_lidar_data(self, msg):
        detected_objects = []
        for i, range_val in enumerate(msg.ranges):
            if 0.0 < range_val < 1.0:  # Object detected if range is within valid bounds
                angle = i % 360  # Ensure angle stays within 0–359°
                detected_objects.append((angle, range_val))  # Store angle and distance
        return detected_objects

    def decide_maneuver(self, detected_objects):
        min_distance = min(detected_objects, key=lambda x: x[1])  # Get the closest object
        angle = min_distance[0]
        distance = min_distance[1]

        # Create a Twist message to send steering commands
        cmd = Twist()
# Debugging output to display detected object's position and distance
        self.get_logger().info(f"Closest object at angle: {angle}, distance: {distance:.2f}m")
        #  logic for maneuvering
        if distance < 0.5:  # Object too close
            if angle < 90 or angle > 270:  # If object is in front
                cmd.angular.z = 1.0  # Turn right
            else:  # Object is behind
                cmd.angular.z = -1.0  # Turn left
        elif 0.5 <= distance < 1.0:  # Object detected but not too close
            if angle < 90:  # Object to the right
                cmd.angular.z = -0.5  # Turn left
            elif angle > 270:  # Object to the left
                cmd.angular.z = 0.5  # Turn right
            else:
                cmd.linear.x = 0.5  # Move forward
        else:  # No close object detected
            cmd.linear.x = 1.0  # Move forward

        # Publish the steering command (Twist message)
        self.steering_publisher.publish(cmd)

    def print_detected_objects(self, detected_objects):
        if detected_objects:
            print("\nDetected Objects:")
            print(f"{'Angle (degrees)':<20} {'Distance (meters)'}")
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
