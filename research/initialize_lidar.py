from adafruit_rplidar import RPLidar
import time

def initialize_lidar(port):
    attempts = 0
    while attempts < 5:
        try:
            lidar = RPLidar(None, port)
            print("Lidar connected successfully.")
            lidar.reset()
            return lidar
        except Exception as e:
            print(f"Failed to initialize LIDAR: {e}")
            if lidar:
                lidar.stop()
                lidar.disconnect()
            time.sleep(1)
            attempts += 1
    raise RuntimeError("Lidar failed to initialize after multiple attempts.")

# Example Usage
if __name__ == "__main__":
    PORT_NAME = '/dev/ttyUSB0'
    lidar = initialize_lidar(PORT_NAME)
