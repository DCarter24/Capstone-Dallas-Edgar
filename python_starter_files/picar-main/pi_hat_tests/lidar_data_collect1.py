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

# Initialize variables
scan_data = [0] * 360
max_distance = 0

# Write the headers in the CSV file
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Angle (Degrees)", "Distance (mm)"])

def process_data(data, timestamp):
    global max_distance
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for angle in range(360):
            distance = data[angle]
            if distance > 0:  # Only write valid distances
                writer.writerow([timestamp, angle, round(distance, 2)])  # Format distance to 2 decimal places

print("Starting RPLidar scan...")
print("Saving data to:", output_file)

try:
    print("Lidar info:", lidar.info)  # Display lidar information
    start_time = time()
    scan_count = 0

    # Main loop: Collect data for 1 minute
    while time() - start_time < 60:  # Run for 60 seconds
        print(f"Scanning... {scan_count + 1}")
        
        for scan in lidar.iter_scans():
            timestamp = datetime.datetime.now().isoformat()
            for (_, angle, distance) in scan:
                scan_data[min([359, floor(angle)])] = distance
            process_data(scan_data, timestamp)
            scan_count += 1
            
            # Display the progress after each scan
            if scan_count % 5 == 0:  # Print progress every 5 scans
                print(f"Processed {scan_count} scans...")

    print("Scan completed.")
    elapsed_time = time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

except KeyboardInterrupt:
    print('Stopping Lidar scan manually...')

finally:
    lidar.stop()
    lidar.disconnect()
    print('Lidar disconnected and program finished.')

# Inform user that data collection is done
print(f"Data collection completed. Data saved to {output_file}.")
