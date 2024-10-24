import math
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
import time
import adafruit_motor.servo

# Hardcoded variables for runtime configuration
run_duration_seconds = 5  # Total duration to run the program (10 seconds)
capture_interval_seconds = 2  # Delay between captures (2 seconds)
debug_mode = True  # Enable or disable debug mode
use_time_limit = True  # Set to True to use time-based limit, False to use specific number of runs
max_runs = 1  # Maximum number of runs if not using time limit

# Motor speed values
neutral_speed = 0  # Neutral (stop) motor position
forward_speed = 0.05  # Slow forward motion
reverse_speed = -0.05  # Slow reverse motion

# Delta timing variables
start_time = time.time()
current_time = start_time
next_run_time = start_time

# Initialize run count
run_count = 0

def Servo_Motor_Initialization():
    """
    Initializes the I2C bus and the PCA9685 motor controller.
    """
    i2c_bus = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c_bus)
    pca.frequency = 100  # Set the frequency of the motor controller
    return pca

def Motor_Start(pca):
    x = input("Press and hold EZ button. Once the LED turns red, immediately relase the button. After the LED blink red once, press 'ENTER'on keyboard.")
    Motor_Speed(pca, 1)
    time.sleep(2)
    y = input("If the LED just blinked TWICE, then press the 'ENTER'on keyboard.")
    Motor_Speed(pca, -1)
    time.sleep(2)
    z = input("Now the LED should be in solid green, indicating the initialization is complete. Press 'ENTER' on keyboard to proceed")
    print("Motor started...")

# Function to control motor speed
def Motor_Speed(pca, percent):
    """
    This function converts a percentage value (-1 to 1) to a 16-bit duty cycle
    and sets the motor speed accordingly.
    """
    # Converts a -1 to 1 value to 16-bit duty cycle
    speed = ((percent) * 3277) + 65535 * 0.15
    pca.channels[15].duty_cycle = math.floor(speed)
    print(f"Motor speed set to {speed/65535:.2f} (16-bit duty cycle)")

# Initialization
pca = Servo_Motor_Initialization()
Motor_Start(pca)

# Main loop
while True:
    # Check if loop should end based on time limit or specific number of runs
    if use_time_limit and (current_time - start_time >= run_duration_seconds):
        break
    elif not use_time_limit and run_count >= max_runs:
        break

    current_time = time.time()

    # Check if it's time for the next run
    if (current_time - next_run_time >= capture_interval_seconds):
        print('____________________Car Motion Started_______________________________')
        next_run_time = current_time
        elapsed_time = current_time - start_time

        print(f"Run Time: {elapsed_time:.2f} seconds")

        # Start driving forward in slow motion
        print("Car is moving forward slowly...")
        Motor_Speed(pca, forward_speed)
        time.sleep(3)  # Run forward for 3 seconds

        # Stop the motor
        print("Car stopping...")
        Motor_Speed(pca, neutral_speed)
        time.sleep(2)  # Stop for 2 seconds

        # Log details
        print(f"Capture Count: {run_count + 1}")

        # Increment run count
        run_count += 1

        # Print debug information if debug mode is enabled
        if debug_mode:
            print(f"Debug: Forward Speed: {forward_speed}, Neutral Speed: {neutral_speed}")
            print(f"Debug: Run Count: {run_count}")

        print('____________________Car Motion Ended_______________________________')

    # Sleep for a short time (this reduces CPU load)
    time.sleep(0.1)

# Stop the motor once the program ends
Motor_Speed(pca, neutral_speed)

print("End of motor control program")
