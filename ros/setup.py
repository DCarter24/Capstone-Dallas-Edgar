from setuptools import setup

package_name = 'rplidar_ros'

setup(
    name=package_name,
    version='1.0',
    packages=[package_name],
    install_requires=['setuptools', 'rclpy', 'sensor_msgs', 'geometry_msgs', 'adafruit-circuitpython-pca9685'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    author='Dallas',  
    author_email='c.dallas@wustl.edu',  
    description='ROS 2 package for Slamtec RPLIDAR with custom steering and LIDAR processing nodes',
    license='washu',  
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'steering_control = rplidar_ros.steering_control:main',
            'lidar_object_detection = rplidar_ros.lidar_object_detection:main',
        ],
    },
)
