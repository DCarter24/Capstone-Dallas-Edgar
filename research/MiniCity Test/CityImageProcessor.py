import matplotlib.pyplot as plt
import numpy as np
import math as m
import time
import cv2

# Initialize camera capture
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define lower and upper HSV limits for the color mask
lower_hsv_limits = (35, 50, 50)  # For green detection
upper_hsv_limits = (85, 255, 255)

# Start image processing
print('____________________Camera Processing Started_______________________________')

# Capture an image using OpenCV
ret, img = cap.read()
if not ret:
    print('ERROR: Failed to Capture Image.')
else:
    # Generate a unique identifier for this image, such as a timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Finding angle using the captured image
    print("_______________________")
    print('___________Finding Angle___________')

    # Basic Printing of Image Characteristics
    print(f'Image Size: {img.size}')
    print(f'Image Shape: {img.shape}')

    # Convert image from RGB to HSV
    print('___________RGB to HSV___________')
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Creating a mask for the specific color range
    print('___________Mask on HSV___________')
    color_mask = cv2.inRange(hsv_image, lower_hsv_limits, upper_hsv_limits)

    # Locating the region of a specific color by finding the moment
    print('___________Moment Def___________')
    moments = cv2.moments(color_mask)
    if moments["m00"] <= 0.00001:
        print(f'ERROR: Moment Calculation Failed: m00 = {moments["m00"]} -- IMAGE IS BLANK/MISSING')
        print(f'Resetting Angle to 360')
        new_angle = 360
    else:
        print('___________Image Detected___________')
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])

        # Create a blurred image
        print('___________Blurred Image___________')
        blurred_mask = cv2.blur(color_mask, (5, 5))

        # Creating the threshold
        threshold_mask = cv2.threshold(blurred_mask, 200, 255, cv2.THRESH_BINARY)[1]

        # Draw a red dot at the center of the detected color region
        print('___________Red Dot Image___________')
        img_with_center_marker = cv2.circle(img, (cX, cY), 5, (0, 0, 255), 2)

        # Save images with unique filenames using the timestamp
        cv2.imwrite(f'center_marker_{timestamp}.png', img_with_center_marker)
        cv2.imwrite(f'mask_{timestamp}.jpg', color_mask)
        cv2.imwrite(f'blurred_image_{timestamp}.jpg', blurred_mask)
        cv2.imwrite(f'threshold_image_{timestamp}.jpg', threshold_mask)
        cv2.imwrite(f'center_marker_image_{timestamp}.jpg', img_with_center_marker)

        # Calculate the angle of the detected region
        print("_______________________")
        print('___________Origin Calc___________')

        # Get the dimensions of the image
        img_height, img_width, _ = img.shape

        # Calculate the reference point (center of the image)
        half_width = img_width / 2

        # Calculate the sides of the triangle formed for angle calculation
        adjacent_side = img_height - cY
        opposite_side = cX - half_width

        # Calculate the angle using arctangent
        angle_theta = m.atan(opposite_side / adjacent_side)
        angle_theta = angle_theta * (180 / m.pi)
        print(f'Theta Found: {angle_theta:.2f}')

        print(f'Pixel X: {cX}, Pixel Y: {cY}')

        # Print debug information
        print('___________Saving Debug Images___________')
        print(f"Center: ({cX}, {cY})")

    # Check the captured angle and update the angle if valid
    if moments["m00"] <= 0.00001:
        print('Second Safety Check: Moment["m00"] <= 0.00001. Angle Set to default 360')
        new_angle = 360
    else:
        new_angle = angle_theta
        print(f'New angle: {new_angle} calculated successfully')

print('____________________Camera Processing Ended_______________________________')

# Release the camera and clean up
cap.release()
print("End of program")
