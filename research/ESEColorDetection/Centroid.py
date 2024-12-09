import cv2
import numpy as np
import logging
import datetime
import math
import sys
import os

# Constants
SCREEN_WIDTH = 640  
SCREEN_HEIGHT = 480 
past_steering_angle = 0
row_threshold = 0
path = "/home/pi/repo/Capstone-Dallas-Edgar/research/ESEColorDetection/PatchData"
crop_height = int(SCREEN_HEIGHT * 0.10)  # This will be 120 pixels
ifblue = False

camera = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)  
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))  # Set YUYV format

def getTime():
    return datetime.datetime.now().strftime("S_%S_M_%M")

def stabilize_steering_angle(curr_steering_angle, last_steering_angle=None, alpha=0.2):
    if last_steering_angle is None:
        return int(curr_steering_angle)
    else:
        if 135 - last_steering_angle <= 5 and curr_steering_angle >= last_steering_angle:
            return np.clip(int(alpha * curr_steering_angle + (1.-alpha) * last_steering_angle),
                           last_steering_angle-1, last_steering_angle+1)
        elif last_steering_angle - 55 <= 5 and curr_steering_angle <= last_steering_angle:
            return np.clip(int(alpha * curr_steering_angle + (1.-alpha) * last_steering_angle),
                           last_steering_angle-1, last_steering_angle+1)
        else:
            return np.clip(int(alpha * curr_steering_angle + (1.-alpha) * last_steering_angle),
                           last_steering_angle-3, last_steering_angle+3)

datestr = getTime()
times2Run = {1}

for i in times2Run:
    for i in times2Run:
        camera.read()  # Discard the first frame
        successfulRead, raw_image = camera.read() 
        if not successfulRead:
            print("Image not taken successful.")
            break

        # Save the raw image immediately after reading for debugging
        cv2.imwrite(os.path.join(path, f"raw_image_{getTime()}.jpg"), raw_image)

        # Validate dimensions to ensure image is read correctly
        if raw_image.shape[1] != SCREEN_WIDTH or raw_image.shape[0] != SCREEN_HEIGHT:
            print(f"Warning: Image dimensions mismatch. Expected: {SCREEN_WIDTH}x{SCREEN_HEIGHT}, Got: {raw_image.shape[1]}x{raw_image.shape[0]}")

        # Flip the raw image
        raw_image = cv2.flip(raw_image, -1)
        cv2.imwrite(os.path.join(path, f"flipped_image_raw_{getTime()}.jpg"), raw_image)
        
        print('Img to color...')
        img_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        print('Cropping top half of the image...')
        # Crop the image 
        img_bottom_half_bgr = raw_image[crop_height:,:]

        print('Performing HSV color space transformation...')
        # Convert only the bottom half to HSV
        img_hsv = cv2.cvtColor(img_bottom_half_bgr, cv2.COLOR_BGR2HSV)
        # Since we already cropped the image by taking only the bottom half,
        # this hsv image is effectively the "cropped hsv". We keep the variable name as requested.
        img_crop_hsv = img_hsv

        print('Creating binary mask...')
        if ifblue:
            lower_hsv = np.array([100, 150, 50])
            upper_hsv = np.array([130, 255, 255])
        else:
            # White detection
            lower_hsv = np.array([0, 0, 120])
            upper_hsv = np.array([180, 50, 255])

        mask = cv2.inRange(img_crop_hsv, lower_hsv, upper_hsv)

        print('Applying Gaussian blur on mask...')
        mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)

        print('Applying Canny filter...')
        mask_edges = cv2.Canny(mask_blurred, 50, 150)

        crop_width = 20  # Adjust this value if needed
        mask_edges = mask_edges[:, crop_width:]

        # Adjust the SCREEN_WIDTH to account for the crop
        adjusted_screen_width = SCREEN_WIDTH - crop_width
        print(f"New width after cropping: {adjusted_screen_width}")

        # Save the cropped mask_edges for debugging
        cv2.imwrite(os.path.join(path, f"cropped_mask_edges_{getTime()}.jpg"), mask_edges)

        minLineLength = 12
        maxLineGap = 3
        min_threshold = 5

        print('Applying Probabilistic Hough Transform...')
        lines = cv2.HoughLinesP(mask_edges, 1, np.pi/180, min_threshold, minLineLength, maxLineGap)

        if lines is not None:
            # Convert mask_edges to BGR for visualization
            hough_debug_img = cv2.cvtColor(mask_edges, cv2.COLOR_GRAY2BGR)

        # Draw the Hough lines on the debug image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
       

        # Save intermediate images
        print("Saving Images without calculating angle.")
        cv2.imwrite(os.path.join(path, f"img_rgb_{getTime()}.jpg"), img_rgb)
        cv2.imwrite(os.path.join(path, f"img_bottom_half_bgr_{getTime()}.jpg"), img_bottom_half_bgr)
        cv2.imwrite(os.path.join(path, f"img_crop_hsv_{getTime()}.jpg"), img_crop_hsv)
        cv2.imwrite(os.path.join(path, f"mask_{getTime()}.jpg"), mask)
        cv2.imwrite(os.path.join(path, f"mask_blurred_{getTime()}.jpg"), mask_blurred)
        cv2.imwrite(os.path.join(path, f"mask_edges_{getTime()}.jpg"), mask_edges)
        cv2.imwrite(os.path.join(path, f"hough_lines_{getTime()}.jpg"), hough_debug_img)

        ### DYNAMIC PATCH GENERATION BASED ON mask_edges ###
        col_sum = np.sum(mask_edges > 0, axis=0)
        threshold = 50  
        lane_columns = np.where(col_sum > threshold)[0]

        if len(lane_columns) == 0:
            list_patch = []
        else:
            segments = []
            start = lane_columns[0]
            prev = lane_columns[0]

            for c in lane_columns[1:]:
                if c != prev + 1:
                    segments.append((start, prev))
                    start = c
                prev = c
            segments.append((start, prev))

            num_patches_vertical = 4  
            patch_height = (SCREEN_HEIGHT - crop_height) // num_patches_vertical
            patch_width = 20  
            list_patch = []

            for (seg_start, seg_end) in segments:
                col_center = (seg_start + seg_end) // 1 + crop_width
                x0 = max(col_center - patch_width//2, 0)
                x1 = min(col_center + patch_width//2, SCREEN_WIDTH - 1)

                for k in range(num_patches_vertical):
                    y0 = k * patch_height
                    y1 = (k+1) * patch_height - 1
                    list_patch.append({'x': (x0, x1), 'y': (y0, y1)})

        # Draw dynamic patches for debugging
        for idx, patch in enumerate(list_patch):
            x0, x1 = patch['x']
            y0, y1 = patch['y']
            cv2.rectangle(img_bottom_half_bgr, (x0, y0), (x1, y1), (0,165,255), 1)
            cv2.rectangle(hough_debug_img, (x0, y0), (x1, y1), (0,165,255), 1)

        print("Saving Image With Lines (Dynamic Patches).")
        cv2.imwrite(os.path.join(path, f"image_lines_bottom_half_raw{getTime()}.jpg"), img_bottom_half_bgr)
        cv2.imwrite(os.path.join(path, f"image_lines_masked_edges{getTime()}.jpg"), hough_debug_img)


        # Centroid Calculation.  
        if lines is None:
            print("No Lines Detected. Exiting Loop")
            break
        else:
            # Create a debug image for centroid visualization
            centroid_debug_image = cv2.cvtColor(hough_debug_img, cv2.COLOR_GRAY2BGR)

            # Data structure to store computed centroids
            patch_centroids_data = []

            # Calculate centroids for each patch based on line data
            for patch_info in list_patch:
                px_start, px_end = patch_info['x']
                py_start, py_end = patch_info['y']

                # Collect all line endpoints that lie fully inside this patch
                inside_points = []
                for detected_line in lines:
                    lx1, ly1, lx2, ly2 = detected_line[0]
                    if (lx1 >= px_start and lx1 <= px_end and ly1 >= py_start and ly1 <= py_end and
                        lx2 >= px_start and lx2 <= px_end and ly2 >= py_start and ly2 <= py_end):
                        inside_points.append([lx1, ly1])
                        inside_points.append([lx2, ly2])

                # If we have endpoints inside this patch, compute a single centroid
                if len(inside_points) > 0:
                    inside_points = np.array(inside_points)
                    centroid_coords = np.mean(inside_points, axis=0)
                    centroid_coords = centroid_coords.astype(int)

                    # Store centroid data for later processing
                    patch_centroids_data.append({'patch': patch_info, 'centroid': (int(centroid_coords[0]), int(centroid_coords[1]))})

                    # Visualize this centroid on the debug image
                    cv2.circle(centroid_debug_image, (int(centroid_coords[0]), int(centroid_coords[1])), 3, (0,165,255), -1)

            # Save the visualization with centroids drawn
            cv2.imwrite(os.path.join(path, f"centroids_visualized_{getTime()}.jpg"), centroid_debug_image)
            print("Centroids computed and visualized on debug image.")
