#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection')
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.threshold_distance = 0.5  # Set a minimum safe distance to objects (in meters)
    ```

# Define the Scan Callback Function
# This function processes incoming LIDAR data, identifies nearby objects, and decides on maneuvers.

    def scan_callback(self, scan_data):
        min_distance = min(scan_data.ranges)  # Get the closest object detected
        if min_distance < self.threshold_distance:
            # Object detected within threshold distance, initiate maneuver
            self.maneuver_around_object()
        else:
            # No obstacle detected, continue moving forward
            self.move_forward()

    def move_forward(self):
        # Command to move the robot forward
        move_cmd = Twist()
        move_cmd.linear.x = 0.2  # Set forward speed
        self.cmd_publisher.publish(move_cmd)

    def maneuver_around_object(self):
        # Command to avoid the obstacle by rotating
        maneuver_cmd = Twist()
        maneuver_cmd.angular.z = 0.5  # Set rotation speed to turn
        self.cmd_publisher.publish(maneuver_cmd)
      
      def main(args=None):
    rclpy.init(args=args)
    object_detection_node = ObjectDetectionNode()
    rclpy.spin(object_detection_node)
    object_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


