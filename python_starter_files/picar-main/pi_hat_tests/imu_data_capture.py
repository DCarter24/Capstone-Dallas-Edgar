import time
import board
import adafruit_mpu6050

i2c = board.I2C()  # uses board.SCL and board.SDA
mpu = adafruit_mpu6050.MPU6050(i2c)

with open("imu_readings.txt", "a") as file:
    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        acceleration = mpu.acceleration
        gyro = mpu.gyro

        log = (f"Timestamp: {timestamp}\n"
               f"Acceleration (m/s^2): X={acceleration[0]:.2f}, Y={acceleration[1]:.2f}, Z={acceleration[2]:.2f}\n"
               f"Gyro (rad/s): X={gyro[0]:.2f}, Y={gyro[1]:.2f}, Z={gyro[2]:.2f}\n\n")

        print(log)
        file.write(log)
        time.sleep(0.1)
