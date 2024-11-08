import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class LidarObjectDetection(Node):
    def __init__(self):
        super().__init__('lidar_object_detection')
        
        # Subscriber to the LIDAR data
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',  # Lidar Topic
            self.lidar_callback,
            10
        )

        # Publisher for steering commands (Twist message for linear and angular velocities)
        self.steering_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
    def lidar_callback(self, msg):
        # Process the LIDAR data to detect objects
        detected_objects = self.process_lidar_data(msg)
        
        # Print detected objects in a nice format
        self.print_detected_objects(detected_objects)
        
        # If any objects are detected, decide on a maneuver
        if detected_objects:
            maneuver = self.decide_maneuver(detected_objects)
            self.publish_steering_command(maneuver)

    def process_lidar_data(self, msg):
       # process the LaserScan data from the LIDAR
        
        detected_objects = []
        for i, range_val in enumerate(msg.ranges):
            if range_val < 1.0:  # Object detected if range is less than 1 meter
                detected_objects.append((i, range_val))  # Store angle and distance

        return detected_objects

    def decide_maneuver(self, detected_objects):
        # Decide the maneuver based on detected objects
        # For simplicity, if an object is on the left or right, we turn accordingly

        min_distance = min(detected_objects, key=lambda x: x[1])  # Get the closest object

        angle = min_distance[0]  # The index is a rough angle in degrees
        if angle < 90:  # Object is to the right
            return 'turn_left'
        else:  # Object is to the left
            return 'turn_right'

    def publish_steering_command(self, maneuver):
        # Publish the steering command based on the maneuver decision
        twist = Twist()
        if maneuver == 'turn_left':
            twist.angular.z = 0.5  # Turn left
        elif maneuver == 'turn_right':
            twist.angular.z = -0.5  # Turn right
        else:
            twist.angular.z = 0.0  # Move straight

        self.steering_publisher.publish(twist)

    def print_detected_objects(self, detected_objects):
        # print data
        if detected_objects:
            print("\nDetected Objects:")
            print(f"{'Angle (degrees)':<20} {'Distance (meters)'}")
            print("-" * 40)
            for obj in detected_objects:
                angle_deg = obj[0]  # LIDAR index is an approximation of the angle in degrees
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
