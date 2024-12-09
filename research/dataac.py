from initialize_lidar import initialize_lidar
import time

def acquire_lidar_data(lidar, attempt=1):
    max_attempts = 3
    try:
        for scan in lidar.iter_scans():
            process_scan_data(scan)
            return  # If data is processed correctly, exit function
    except Exception as e:
        print(f"Attempt {attempt}: Error during data acquisition: {e}")
        if attempt < max_attempts:
            print(f"Attempting to reset LIDAR... (Attempt {attempt + 1} of {max_attempts})")
            lidar.reset()
            time.sleep(2)  # wait a bit before retrying
            acquire_lidar_data(lidar, attempt + 1)  # Recursive retry
        else:
            print("Max attempts reached. Check hardware connections and settings.")
            raise  # Re-raise the last exception after retry limits
45

def process_scan_data(scan):
    # Implement the logic to process each scan
    print("Processing scan data...")

# Example Usage
if __name__ == "__main__":
    PORT_NAME = '/dev/ttyUSB0'
    lidar = initialize_lidar(PORT_NAME)
    while True:
        acquire_lidar_data(lidar)
