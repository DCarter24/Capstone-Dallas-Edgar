import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
from board import SCL, SDA
import busio
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685

class SteeringControl(Node):
    def __init__(self):
        super().__init__('steering_control')

        # Subscriber to the /cmd_vel topic for steering commands
        self.steering_subscription = self.create_subscription(
            Twist,
            '/cmd_vel',  # The topic where the object detection file publishes the commands
            self.steering_callback,
            10
        )

        # Initialize PCA9685 and servo
        self.i2c = busio.I2C(SCL, SDA)
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = 100
        self.servo7 = servo.Servo(self.pca.channels[14])

    def steering_callback(self, msg):
        # Callback function to handle the steering command
        angular_velocity = msg.angular.z
        if angular_velocity > 0:  # Turn right
            self.turn_right()
        elif angular_velocity < 0:  # Turn left
            self.turn_left()
        else:  # Go straight
            self.straight()

    def turn_left(self):
        self.get_logger().info('Turning left')
        self.servo7.angle = 45  # Adjust the angle as needed
        time.sleep(0.5)

    def turn_right(self):
        self.get_logger().info('Turning right')
        self.servo7.angle = 135  # Adjust the angle as needed
        time.sleep(0.5)

    def straight(self):
        self.get_logger().info('Moving straight')
        self.servo7.angle = 90  # Default straight position
        time.sleep(0.5)

def main(args=None):
    rclpy.init(args=args)
    steering_control = SteeringControl()
    rclpy.spin(steering_control)
    steering_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
