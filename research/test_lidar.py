import math
import os
import time
import busio
from math import cos, sin, pi, floor
from board import SCL, SDA
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from adafruit_rplidar import RPLidar
import trialmotor as momo

i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 100

os.putenv('SDL_FBDEV', '/dev/fb1')
PORT_NAME = '/dev/ttyUSB0'
lidar = RPLidar(None, PORT_NAME, timeout=3)
steering_channel = 14
motor_channel = 15
servo_steering = servo.Servo(pca.channels[steering_channel])

def update_steering_angle(angle):
    servo_steering.angle = angle
    
def scale_lidar_distance(distance, max_distance=3000):
    return min(distance, max_distance) / max_distance

def main():
    # neutral
    update_steering_angle(97)
    time.sleep(0.1)
    momo.Motor_Speed(pca,0.175)
    count = 0
    reset = False
    
    try:
        scan_data = [0]*360
        while True:
            for scan in lidar.iter_scans():
                for (_, angle, distance) in scan:
                    angle = int(angle)
                    
                    # Print Out Data
                    f_dist = f"{distance:.2f}"
                    f_angle = f"{angle:.2f}"
                    print(f"D: {f_dist} mm, A: {f_angle} deg")
                    
                    # Back
                    if distance <= 225 and (angle in range(315, 360) or angle in range(0,45)):
                        print("Object Behind!")
                        momo.Motor_Speed(pca,0)
                        exit()
                        
                    # Front
                    if distance <= 675 and (angle in range(180, 230)):
                        print("Object Front!")
                        momo.Motor_Speed(pca,0)
                        exit()
                        
                    # Left
                    if distance <= 650 and (angle in range(45, 180)):
                        print("Open space on the left side!")
                        update_steering_angle(60)
                        
                    # Right
                    if distance <= 650 and (angle in range(230, 315)):
                        print("Open space on the right side!")
                        update_steering_angle(130)
                                            
                # LIDAR scaling
                for angle in range(360):
                    distance = scan_data[angle]
                    if distance:
                        scaled_distance = scale_lidar_distance(distance)
                        radians = angle * pi / 180
                        x = scaled_distance * cos(radians) * 119
                        y = scaled_distance * sin(radians) * 119
                        point = (160 + int(x), 120 + int(y))
                        
    except KeyboardInterrupt:
        print('Stopping.')
    finally:
        lidar.stop()
        lidar.disconnect()
if __name__ == "__main__":
    main()
