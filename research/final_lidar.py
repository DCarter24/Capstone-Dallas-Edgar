#!/usr/bin/env python3 

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
import time  
from board import SCL, SDA
import busio
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685

class LidarObjectDetection(Node):
    
    def Servo_Motor_Initialization():
        i2c_bus = busio.I2C(SCL, SDA)
        pca = PCA9685(i2c_bus)
        pca.frequency = 100
        return pca
    def Motor_Speed(self, percent):
        # Convert a -1 to 1 value to a 16-bit duty cycle
        speed = ((percent) * 3277) + 65535 * 0.15
        pca.channels[15].duty_cycle = math.floor(speed)
        #self.get_logger().info(f'Motor Speed: {speed / 65535:.2f}')
        print(speed/65535)
        
    def __init__(self):
        super().__init__('lidar_object_detection')

        # Initialize servo motor for both steering and speed control
        self.pca = self.Servo_Motor_Initialization()
        self.steering_servo = servo.Servo(self.pca.channels[14])  
        self.speed_servo = servo.Servo(self.pca.channels[15])     
        self.pca.frequency = 100

        # Set initial positions for servo motors CHECK HERE FOR ERRORS
        self.steering_servo.angle = 90  # Neutral steering (straight)
        self.speed_servo.angle = 90     # Neutral speed (stopped)

        # Initialize LIDAR subscriber
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

    def lidar_callback(self, msg):
        # Process LIDAR scan data to detect objects
        detected_objects = self.process_lidar_data(msg)

        if detected_objects:
            # Select the closest object
            closest_object = min(detected_objects, key=lambda x: x[1])
            angle, distance = closest_object
            self.get_logger().info(f'Object detected at angle {angle}° and distance {distance:.2f}m')
            self.decide_maneuver(angle, distance)
        else:
            # No objects detected, move straight
            self.get_logger().info('No objects detected, moving straight.')
            self.move_forward()

    def process_lidar_data(self, msg):
        detected_objects = []
        for i, distance in enumerate(msg.ranges):
            if 0.1 < distance < 2.0:  # Filter valid range (e.g., 10cm to 2m)
                angle = math.degrees(msg.angle_min + i * msg.angle_increment)
                if 0 <= angle <= 180:  # Consider objects in front
                    detected_objects.append((angle, distance))
        return detected_objects

    def decide_maneuver(self, angle, distance):
        # Control logic based on object position and proximity
        if 180 < angle < 270:  # Object is on the right, FIX THIS
            self.get_logger().info('Object on the right, turning left.')
            self.turn_right()  
        elif 360> angle > 271:  # Object is on the left, FIX THIS
            self.get_logger().info('Object on the left, turning right.')
            self.turn_left()  
        else:  # Object directly ahead
            if distance < 0.01:  # Too close
                self.get_logger().info('Object directly ahead, stopping.')
                self.stop()
            else:
                self.get_logger().info('Object directly ahead, moving forward cautiously.')
                self.move_forward_slow()

    def move_forward(self):
        self.get_logger().info('Moving forward.')
        self.Motor_Speed(pca, 0.15)  # Forward speed
        self.steering_servo.angle = 90  # Keep steering straight
        time.sleep(0.3)  

    def move_forward_slow(self):
        self.get_logger().info('Moving forward slowly.')
        self.Motor_Speed(pca, 0.05)  # Slower forward speed
        self.steering_servo.angle = 90  # Keep steering straight
        time.sleep(0.3)  

    def turn_left(self):
        self.get_logger().info('Turning left.')
        self.steering_servo.angle = 30  # Turn left (servo angle adjusted)
        self.Motor_Speed(pca, 0.05)  # Slow down while turning (adjust)
        time.sleep(0.3)  

    def turn_right(self):
        self.get_logger().info('Turning right.')
        self.steering_servo.angle = 120  # Turn right (servo angle adjusted)
        self.Motor_Speed(pca, 0.15)  # Slow down while turning (adjust)
        time.sleep(0.3)  

    def stop(self):
        self.get_logger().info('Stopping.')
        self.Motor_Speed(pca, 0)  # Stop motor
        time.sleep(0.3)  

def main(args=None):
    rclpy.init(args=args)
    lidar_object_detection = LidarObjectDetection()
    rclpy.spin(lidar_object_detection)
    lidar_object_detection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
