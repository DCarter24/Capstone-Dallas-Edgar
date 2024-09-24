import os
from math import cos, sin, pi, floor
from time import time, sleep
from adafruit_rplidar import RPLidar
import csv
import datetime

# Setup the RPLidar
PORT_NAME = '/dev/ttyUSB0'
lidar = RPLidar(None, PORT_NAME, timeout=3)

# File to store the data
output_file = 'lidar_data.csv'

# Write the headers in the CSV file
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Angle", "Distance"])

# used to scale data to fit on the screen
max_distance = 0

def process_data(data, timestamp):
    global max_distance
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for angle in range(360):
            distance = data[angle]
            if distance > 0:  # ignore initially ungathered data points
                writer.writerow([timestamp, angle, distance])

scan_data = [0] * 360

try:
    print(lidar.info)
    start_time = time()
    while time() - start_time < 60:  # Run for one minute
        for scan in lidar.iter_scans():
            timestamp = datetime.datetime.now().isoformat()
            for (_, angle, distance) in scan:
                scan_data[min([359, floor(angle)])] = distance
            process_data(scan_data, timestamp)

except KeyboardInterrupt:
    print('Stopping.')
finally:
    lidar.stop()
    lidar.disconnect()
