import os
from math import cos, sin, pi, floor
from adafruit_rplidar import RPLidar

# Setup the RPLidar
PORT_NAME = '/dev/ttyUSB0'
lidar = RPLidar(None, PORT_NAME, timeout=3)

# Define a proximity threshold for obstacle avoidance in centimeters
PROXIMITY_THRESHOLD = 50  # adjust
SCAN_ANGLE = 30  # Angle range (e.g., -15 to +15 degrees) to check in front

def process_data(data):
    global max_distance
    obstacle_detected = False
    left_obstacle = False
    right_obstacle = False
    
    for angle in range(360):
        distance = data[angle]
        
        if distance > 0:  # Ignore initially ungathered data points
            radians = angle * pi / 180.0
            x = distance * cos(radians)
            y = distance * sin(radians)
            
            # Check if obstacle is within proximity threshold in the front area
            if distance < PROXIMITY_THRESHOLD:
                if -SCAN_ANGLE <= angle <= SCAN_ANGLE:
                    obstacle_detected = True
                    if angle < 180:  # Object on the left side
                        left_obstacle = True
                    else:  # Object on the right side
                        right_obstacle = True
    
    if obstacle_detected:
        maneuver(left_obstacle, right_obstacle)

def maneuver(left_obstacle, right_obstacle):
    if left_obstacle and right_obstacle:
        print("Obstacle ahead! Stop or reverse.")
        # Code to stop or reverse the iCar
    elif left_obstacle:
        print("Obstacle on the left. Turning right.")
        # Code to turn right
    elif right_obstacle:
        print("Obstacle on the right. Turning left.")
        # Code to turn left
    else:
        print("Clear path ahead.")
        # Code to move forward

scan_data = [0] * 360

try:
    print(lidar.info)
    for scan in lidar.iter_scans():
        for (_, angle, distance) in scan:
            scan_data[min([359, floor(angle)])] = distance
        process_data(scan_data)

except KeyboardInterrupt:
    print('Stopping.')
lidar.stop()
lidar.disconnect()
