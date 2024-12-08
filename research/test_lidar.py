#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
from board import SCL, SDA
import busio
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685

class LidarObjectDetection(Node):
    NEUTRAL_ANGLE = 90
    SLOW_SPEED_PERCENT = 0.15
    TURN_SPEED_PERCENT = 0.2
    STOP_SPEED = 0
    DETECTION_RANGE_MIN = 0.1  # in meters
    DETECTION_RANGE_MAX = 2.0  # in meters
    RIGHT_TURN_ANGLE = 135
    LEFT_TURN_ANGLE = 45

    def __init__(self):
        super().__init__('lidar_object_detection')
        self.pca = self.init_servo_motor()
        self.steering_servo = servo.Servo(self.pca.channels[14])
        self.pca.frequency = 100
        self.steering_servo.angle = self.NEUTRAL_ANGLE
        self.lidar_subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)

    def init_servo_motor(self):
        i2c_bus = busio.I2C(SCL, SDA)
        pca = PCA9685(i2c_bus)
        pca.frequency = 100
        return pca

    def motor_speed(self, percent):
        speed = int((percent * 32767) + 65535 * 0.15)
        self.pca.channels[15].duty_cycle = speed
        self.get_logger().info(f'Motor Speed: {percent:.2f}')

    def lidar_callback(self, msg):
        objects = [(i, distance) for i, distance in enumerate(msg.ranges) if self.DETECTION_RANGE_MIN < distance < self.DETECTION_RANGE_MAX]
        if objects:
            closest_object = min(objects, key=lambda x: x[1])
            angle = math.degrees(msg.angle_min + closest_object[0] * msg.angle_increment)
            self.decide_maneuver(angle, closest_object[1])
        else:
            self.get_logger().info('No objects detected, moving straight.')
            self.move_forward()

    def decide_maneuver(self, angle, distance):
        if angle <= 45 or angle >= 315:
            self.handle_ahead_object(angle, distance)
        elif 45 < angle <= 90:
            self.turn_left()
        elif 270 < angle < 315:
            self.turn_right()

    def handle_ahead_object(self, angle, distance):
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

def main(args=None):
    rclpy.init(args=args)
    lidar_object_detection = LidarObjectDetection()
    rclpy.spin(lidar_object_detection)
    lidar_object_detection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
