import time
import board
import adafruit_mpu6050

# Initialize IMU sensor
i2c = board.I2C()
mpu = adafruit_mpu6050.MPU6050(i2c)

# Variables for distance calculation
velocity = [0, 0, 0]  # Velocity components in X, Y, and Z directions
position = [0, 0, 0]  # Position components in X, Y, and Z directions

# Constants
alpha = 0.8  # Smoothing factor for the low-pass filter (between 0 and 1)
calibration_interval = 10  # Interval for recalibrating the gravity vector (in seconds)

# Initialize filtered acceleration values (for low-pass filtering)
filtered_acceleration = [0, 0, 0]
gravity_vector = [0, 0, 0]  # Placeholder for the dynamic gravity vector

# Perform initial calibration phase to determine the gravitational vector
def calibrate_gravity_vector():
    print("Calibrating gravity vector...")
    calibration_duration = 5  # Collect data for 5 seconds for calibration
    calibration_samples = 0
    temp_gravity_vector = [0, 0, 0]

    calibration_start_time = time.time()
    while time.time() - calibration_start_time < calibration_duration:
        raw_acceleration = mpu.acceleration
        for i in range(3):
            temp_gravity_vector[i] += raw_acceleration[i]
        calibration_samples += 1
        time.sleep(0.1)  # Collect samples at 10 Hz

    # Average the collected acceleration values to determine the gravity vector
    for i in range(3):
        gravity_vector[i] = temp_gravity_vector[i] / calibration_samples

    print(f"Gravity vector calibrated: X={gravity_vector[0]:.2f}, Y={gravity_vector[1]:.2f}, Z={gravity_vector[2]:.2f}")

# Initial gravity calibration
calibrate_gravity_vector()

# Main part of the program
with open("imu_readings.txt", "a") as file:
    while True:
        print('____________________IMU Data Collection Started_______________________________')

        # Read current acceleration data
        raw_acceleration = mpu.acceleration

        # Apply a low-pass filter to smooth out noise in the raw acceleration data
        for i in range(3):
            filtered_acceleration[i] = alpha * filtered_acceleration[i] + (1 - alpha) * raw_acceleration[i]
            # Compensate for gravity by subtracting the dynamic gravity vector
            filtered_acceleration[i] -= gravity_vector[i]

        # Calculate distance using kinematic equations for each axis
        delta_t = 0.1  # Fixed delta time of 0.1 seconds (10 Hz)
        for i in range(3):  # Loop over X, Y, Z axes
            # Update velocity: v_new = v_old + a * delta_t
            velocity[i] += filtered_acceleration[i] * delta_t

            # Update position: distance = 0.5 * a * (delta_t^2) + v_old * delta_t
            position[i] += 0.5 * filtered_acceleration[i] * (delta_t ** 2) + velocity[i] * delta_t

        # Log and print current IMU data
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log = (
            f"Timestamp: {timestamp}\n"
            f"Raw Acceleration (m/s^2): X={raw_acceleration[0]:.2f}, Y={raw_acceleration[1]:.2f}, Z={raw_acceleration[2]:.2f}\n"
            f"Filtered Acceleration (m/s^2): X={filtered_acceleration[0]:.2f}, Y={filtered_acceleration[1]:.2f}, Z={filtered_acceleration[2]:.2f}\n"
            f"Velocity (m/s): X={velocity[0]:.2f}, Y={velocity[1]:.2f}, Z={velocity[2]:.2f}\n"
            f"Position (m): X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f}\n\n"
        )
        print(log)
        file.write(log)
        file.flush()  # Ensure data is written immediately

        # Print debug information if debug mode is enabled
        if debug_mode:
            print(f'Debug: Filtered Acceleration X={filtered_acceleration[0]:.2f}, Y={filtered_acceleration[1]:.2f}, Z={filtered_acceleration[2]:.2f}')
            print(f'Debug: Velocity X={velocity[0]:.2f}, Y={velocity[1]:.2f}, Z={velocity[2]:.2f}')
            print(f'Debug: Position X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f}')

        print('____________________IMU Data Collection Ended_______________________________')

        # Sleep for 0.1 seconds (10 Hz)
        time.sleep(0.1)

print("End of IMU Data Collection")
