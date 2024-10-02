import matplotlib.pyplot as plt
import numpy as np
import math as m
import time
import cv2

# Hardcoded variables for runtime configuration
run_duration_seconds = 60  # Total duration to run the program (1 minute)
capture_interval_seconds = 5  # Delay between captures (5 seconds)
debug_mode = True  # Enable or disable debug mode

# Initialize camera capture
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define lower and upper HSV limits for the color mask
lower_hsv_limits = (35, 50, 50) # This represents a green hue with medium to high saturation and brightness.
upper_hsv_limits = (85, 255, 255) # This upper limit encompasses the hue range for green with maximum saturation and brightness.

# Initialize timing variables
start_time = time.time()
current_time = start_time
next_capture_time = start_time

# Main part of the program
while (current_time - start_time < run_duration_seconds):
    current_time = time.time()

    if (current_time - next_capture_time >= capture_interval_seconds):
        print('____________________Camera Processing Started_______________________________')
        next_capture_time = current_time
        elapsed_capture_time = current_time - start_time

        print(f'Capture Time: {elapsed_capture_time:.2f} seconds')

        # Capture an image using OpenCV
        ret, img = cap.read()
        if not ret:
            print(f'ERROR: Failed to Capture Image. Time Elapsed: {current_time:.2f}')
            continue  # Skip this iteration if the image was not captured successfully

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
            print(f' Resetting Angle to 360')
            new_angle = 360
            print(f' Time Elapsed: {current_time:.2f}')
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
            cv2.imwrite('center_marker.png', img_with_center_marker)

            # Calculate the angle of the detected region
            print("_______________________")
            print('___________Origing Calc___________')

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
            print(f'Theta Found: {angle_theta:.2f} at {current_time:.2f} seconds')

            print(f'Pixel X: {cX}, Pixel Y: {cY} at {current_time:.2f} seconds')

            if debug_mode:
                # Save debug images
                print('___________Saving Debug Images___________')
                cv2.imwrite('mask.jpg', color_mask)
                cv2.imwrite('blurred_image.jpg', blurred_mask)
                cv2.imwrite('threshold_image.jpg', threshold_mask)
                cv2.imwrite('center_marker_image.jpg', img_with_center_marker)

                print(f"Center: ({cX}, {cY})")

        # Check the captured angle and update the angle if valid
        if moments["m00"] <= 0.00001:
            print("SecondSafety Check: Moment[\"m00\"] <= 0.00001. Angle Set to default 360")
            new_angle = 360
        else:
            new_angle = angle_theta
            print(f'New angle: {new_angle} calculated at runtime: {current_time:.2f}')

        print('____________________Camera Processing Ended_______________________________')

# Release the camera and clean up
cap.release()
car.stop()
print("End of program")
