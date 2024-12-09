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
times2Run = {1}


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
            col_center = (seg_start + seg_end) // 2 + crop_width
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
        cv2.rectangle(img_bottom_half_bgr, (x0, y0), (x1, y1), (50,255, 0), 1)
        cv2.rectangle(hough_debug_img, (x0, y0), (x1, y1), (50,255, 0), 1)

    print("Saving Image With Lines (Dynamic Patches).")
    cv2.imwrite(os.path.join(path, f"image_lines_bottom_half_raw{getTime()}.jpg"), img_bottom_half_bgr)
    cv2.imwrite(os.path.join(path, f"image_lines_masked_edges{getTime()}.jpg"), hough_debug_img)


    # Centroid Calculation.  
    if lines is None:
        print("No Lines Detected. Exiting Loop")
        break
    else:
        # Create a debug image for centroid visualization
        centroid_debug_image = hough_debug_img.copy()

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

        # Create arrays for left and right centroids
        X_left = []
        X_right = []
        # In case we need directional counters (example code used these)
        n_right_side_right_dir = 0
        n_right_side_left_dir = 0
        n_left_side_right_dir = 0
        n_left_side_left_dir = 0

        # Separate centroids into left or right side
        # For now, we will not use velocity as it's not defined here.
        # We'll assume that all centroids with x < 160 belong to the left side,
        # and all centroids with x >= 160 belong to the right side.
        # This is a simplification to illustrate the idea.
        for data_item in patch_centroids_data:
            cx, cy = data_item['centroid']  # centroid is (x, y)
            if cx < 160:
                X_left.append([cx, cy])
            else:
                X_right.append([cx, cy])

        # Convert to numpy arrays if we have points
        X_left = np.array(X_left) if len(X_left) > 0 else np.zeros((0,2))
        X_right = np.array(X_right) if len(X_right) > 0 else np.zeros((0,2))

        # Polynomial interpolation
        # We'll attempt a simple linear fit y = m*x + b
        # Only proceed if we have at least 2 points on each side
        # Visualization will be done on hough_debug_img which already has patches
        poly_debug_img = hough_debug_img.copy()

        x_start_right = None
        x_start_left = None

        # Fit line on right side
        if len(X_right) > 1:
            print('Computing polynomial interpolation for right lane...')
            # Fit a polynomial (linear)
            right_lane = np.polyfit(X_right[:,0], X_right[:,1], 1, w=X_right[:,1])
            # Let's draw a line between two x-coordinates, say from x=219 to x=319 as per example
            y1 = right_lane[0] * 219 + right_lane[1]
            y2 = right_lane[0] * 319 + right_lane[1]
            cv2.line(poly_debug_img, (219,int(y1)), (319,int(y2)), (0,255,255), 5)
            x_start_right = int((25 - right_lane[1])/(right_lane[0]+0.001))

        # Fit line on left side
        if len(X_left) > 1:
            print('Computing polynomial interpolation for left lane...')
            left_lane = np.polyfit(X_left[:,0], X_left[:,1], 1, w=X_left[:,1])
            # Draw a line from x=0 to x=100 as per example
            y1 = left_lane[0] * 0 + left_lane[1]
            y2 = left_lane[0] * 100 + left_lane[1]
            cv2.line(poly_debug_img, (0,int(y1)), (100,int(y2)), (0,255,255), 5)
            x_start_left = int((25 - left_lane[1])/(left_lane[0]+0.001))

        # Save the image with polynomial lines
        cv2.imwrite(os.path.join(path, f"polynomial_lines_{getTime()}.jpg"), poly_debug_img)
        print("Polynomial lines computed and visualized.")

        # Steering angle calculation
        # We'll compute mid_star based on whether we have both lines, one line, or none
        if (x_start_right is not None) and (x_start_left is not None):
            mid_star = 0.5 * (x_start_right + x_start_left)
            cv2.line(poly_debug_img,(int(np.clip(mid_star,-10000,10000)),25),(160,100),(255,0,255),5)
        elif (x_start_right is not None) and (x_start_left is None):
            # Only right line available
            mid_star = (25-100)/right_lane[0] + 160
            cv2.line(poly_debug_img,(int(np.clip(mid_star,-10000,10000)),25),(160,100),(255,0,255),5)
        elif (x_start_right is None) and (x_start_left is not None):
            # Only left line available
            mid_star = (25-100)/left_lane[0] + 160
            cv2.line(poly_debug_img,(int(np.clip(mid_star,-10000,10000)),25),(160,100),(255,0,255),5)
        else:
            # No lines
            mid_star = 159

        # Compute steering angle as per example logic
        print('Computing steering angle...')
        if np.abs(mid_star-160)<2:
            steering_angle = 90
        else:
            steering_angle = 90 + np.degrees(np.arctan((mid_star-160)/75.))
            steering_angle = np.clip(steering_angle,55,135)

        # Stabilize angle
        stable_steering_angle = stabilize_steering_angle(steering_angle,past_steering_angle)

        # Add steering angle text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(stable_steering_angle)
        cv2.putText(poly_debug_img, text, (110, 30 ), font, 1, (0, 0, 255), 2)

        # Before concatenating, ensure poly_debug_img and raw_image[:crop_height,:] have the same width
        top_section = raw_image[:crop_height,:]
        top_h, top_w, _ = top_section.shape
        poly_h, poly_w, _ = poly_debug_img.shape

        # If widths differ, pad poly_debug_img to match top_w
        if poly_w < top_w:
            diff = top_w - poly_w
            poly_debug_img = cv2.copyMakeBorder(poly_debug_img, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))

        # Now concatenate the top raw portion with the processed bottom portion
        new_frame = np.concatenate((top_section, poly_debug_img), axis=0)

        # Draw steering line on new_frame
        height, width, _ = new_frame.shape
        start_point = (int(width / 2), int(height))
        angle_from_vertical = stable_steering_angle - 90
        angle_rad = np.radians(angle_from_vertical)
        line_length = 100  
        end_point_x = int(start_point[0] + line_length * np.sin(angle_rad))
        end_point_y = int(start_point[1] - line_length * np.cos(angle_rad))
        end_point_x = max(0, min(end_point_x, width - 1))
        end_point_y = max(0, min(end_point_y, height - 1))
        end_point = (end_point_x, end_point_y)
        cv2.line(new_frame, start_point, end_point, (255, 0, 255), thickness=2)

        # Save image with steering angle
        cv2.imwrite(os.path.join(path, f"final_frame_image_{getTime()}.jpg"), new_frame)
        print("Steering angle computed and visualized.")
        print("End.")
