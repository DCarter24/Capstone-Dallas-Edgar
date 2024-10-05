import time
import board
import adafruit_mpu6050

# Hardcoded variables for runtime configuration
run_duration_seconds = 60  # Total duration to run the program (1 minute)
capture_interval_seconds = 3  # Delay between captures (5 seconds)
debug_mode = True  # Enable or disable debug mode
use_time_limit = False  # Set to True to use time-based limit, False to use capture count
max_captures = 5  # Maximum number of captures if not using time limit

# Initialize IMU sensor
i2c = board.I2C()
mpu = adafruit_mpu6050.MPU6050(i2c)

# Variables for distance calculation
velocity = [0, 0, 0]  # Velocity components in X, Y, and Z directions
position = [0, 0, 0]  # Position components in X, Y, and Z directions

# Initialize timing variables
start_time = time.time()
current_time = start_time
next_capture_time = start_time

# Initialize capture count
capture_count = 0

# Main part of the program
with open("imu_readings.txt", "a") as file:
    while True:
        # Check if the loop should end based on time limit or capture count
        if use_time_limit and (current_time - start_time >= run_duration_seconds):
            break
        elif not use_time_limit and capture_count >= max_captures:
            break

        current_time = time.time()

        if (current_time - next_capture_time >= capture_interval_seconds):
            print('____________________IMU Data Collection Started_______________________________')
            next_capture_time = current_time
            elapsed_capture_time = current_time - start_time

            print(f'Capture Time: {elapsed_capture_time:.2f} seconds')

            # Read current acceleration and gyro data
            acceleration = mpu.acceleration
            delta_t = current_time - start_time

            # Calculate distance using kinematic equations for each axis
            for i in range(3):  # Loop over X, Y, Z axes
                # Update velocity: v_new = v_old + a * delta_t
                velocity[i] += acceleration[i] * delta_t

                # Update position: distance = 0.5 * a * (delta_t^2) + v_old * delta_t
                position[i] += 0.5 * acceleration[i] * (delta_t ** 2) + velocity[i] * delta_t

            # Log and print current IMU data
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log = (
                f"Timestamp: {timestamp}\n"
                f"Acceleration (m/s^2): X={acceleration[0]:.2f}, Y={acceleration[1]:.2f}, Z={acceleration[2]:.2f}\n"
                f"Velocity (m/s): X={velocity[0]:.2f}, Y={velocity[1]:.2f}, Z={velocity[2]:.2f}\n"
                f"Position (m): X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f}\n\n"
            )
            print(log)
            file.write(log)
            file.flush()  # Ensure data is written immediately

            # Increment capture count
            capture_count += 1
            print(f'Capture Count: {capture_count}')

            # Print debug information if debug mode is enabled
            if debug_mode:
                print(f'Debug: Velocity X={velocity[0]:.2f}, Y={velocity[1]:.2f}, Z={velocity[2]:.2f}')
                print(f'Debug: Position X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f}')

            print('____________________IMU Data Collection Ended_______________________________')

        # Sleep for 0.1 seconds (10 Hz)
        time.sleep(0.1)

print("End of IMU Data Collection")
