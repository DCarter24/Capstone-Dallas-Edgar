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
crop_height = int(SCREEN_HEIGHT * 0.25)  # This will be 120 pixels
ifblue = False

camera = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)  
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

def getTime():
    return datetime.datetime.now().strftime("%y%m%d_%H%M%S")

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
times2Run = {2}

for i in times2Run:
    for i in times2Run:
        successfulRead, raw_image = camera.read() 
        if not successfulRead:
            print("Image not taken successful.")
            break

        white_bar_width = 10
        
        raw_image = raw_image[:, white_bar_width:]
        adjusted_screen_width = SCREEN_WIDTH - white_bar_width
        row_threshold = SCREEN_HEIGHT - crop_height
        raw_image = cv2.flip(raw_image, -1)
        
        print('Img to color...')
        img_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        print('Cropping top half of the image...')

        # Convert the cropped portion to BGR directly, without additional slicing
        img_bottom_half_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)


        print('Performing HSV color space transformation...')
        img_hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
        img_top_half_bgr = raw_image[:crop_height,:]
        img_crop_hsv = img_hsv[crop_height:,:]

        print('Creating binary mask...')
        if ifblue:
            lower_hsv = np.array([100, 150, 50])
            upper_hsv = np.array([130, 255, 255])
        else:
            # White detection
            lower_hsv = np.array([0, 0, 200])
            upper_hsv = np.array([180, 25, 255])

        mask = cv2.inRange(img_crop_hsv, lower_hsv, upper_hsv)

        print('Applying Gaussian blur on mask...')
        mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)

        print('Applying Canny filter...')
        mask_edges = cv2.Canny(mask_blurred, 50, 150)

        minLineLength = 12
        maxLineGap = 3
        min_threshold = 5

        print('Applying Probabilistic Hough Transform...')
        lines = cv2.HoughLinesP(mask_edges, 1, np.pi/180, min_threshold, minLineLength, maxLineGap)

        # Save intermediate images
        print("Saving Images without calculating angle.")
        cv2.imwrite(os.path.join(path, f"img_rgb_{getTime()}.jpg"), img_rgb)
        cv2.imwrite(os.path.join(path, f"img_bottom_half_bgr_{getTime()}.jpg"), img_bottom_half_bgr)
        cv2.imwrite(os.path.join(path, f"img_crop_hsv_{getTime()}.jpg"), img_crop_hsv)
        cv2.imwrite(os.path.join(path, f"mask_{getTime()}.jpg"), mask)
        cv2.imwrite(os.path.join(path, f"mask_blurred_{getTime()}.jpg"), mask_blurred)
        cv2.imwrite(os.path.join(path, f"mask_edges_{getTime()}.jpg"), mask_edges)

        ### DYNAMIC PATCH GENERATION BASED ON mask_edges ###
        # Sum of white pixels per column to find potential lane locations
        col_sum = np.sum(mask_edges > 0, axis=0)
        # Set a threshold to identify columns with a significant amount of edges
        # This threshold might need tuning depending on lighting and lane visibility.
        threshold = 50  
        lane_columns = np.where(col_sum > threshold)[0]

        if len(lane_columns) == 0:
            # No significant columns found - fallback to no patches or default logic
            list_patch = []
        else:
            # Group adjacent columns to form segments
            segments = []
            start = lane_columns[0]
            prev = lane_columns[0]

            for c in lane_columns[1:]:
                if c != prev + 1:
                    segments.append((start, prev))
                    start = c
                prev = c
            segments.append((start, prev))

            # Define patch parameters
            # For example, we create a few vertical patches per lane segment.
            # Adjust as needed.
            num_patches_vertical = 4  
            patch_height = (SCREEN_HEIGHT - crop_height) // num_patches_vertical
            patch_width = 20  # width of each patch
            list_patch = []

            for (seg_start, seg_end) in segments:
                col_center = (seg_start + seg_end) // 2
                # Center patch around col_center
                x0 = max(col_center - patch_width//2, 0)
                x1 = min(col_center + patch_width//2, SCREEN_WIDTH - 1)

                # Create vertical patches
                for k in range(num_patches_vertical):
                    y0 = k * patch_height
                    y1 = (k+1)*patch_height - 1
                    list_patch.append({'x': (x0, x1), 'y': (y0, y1)})

        # Draw dynamic patches for debugging
        for idx, patch in enumerate(list_patch):
            x0, x1 = patch['x']
            y0, y1 = patch['y']
            cv2.rectangle(img_bottom_half_bgr, (x0, y0), (x1, y1), (0,165,255), 1)

        print("Saving Image With Lines (Dynamic Patches).")
        cv2.imwrite(os.path.join(path, f"image_lines{getTime()}.jpg"), img_bottom_half_bgr)

        if lines is None:
            print("No Lines Detected. Exiting Loop")
            break
        else:
            X_left = np.zeros((1,2))
            X_right = np.zeros((1,2))
            n_right_side_right_dir = 0
            n_right_side_left_dir = 0
            n_left_side_right_dir = 0
            n_left_side_left_dir = 0

            print('Computing centroids and mean direction...')
            for patch in list_patch: 
                centroids = {'bottom': np.zeros((1,2)),'top': np.zeros((1,2))}
                velocity = (None,None)
                empty_bool = True

                for line in lines:
                    x1,y1,x2,y2 = line[0]
                    # Check if line endpoints are inside this patch
                    if x1>=patch['x'][0] and x1<=patch['x'][1] and y1<=patch['y'][1] and y1>=patch['y'][0]:
                        if x2>=patch['x'][0] and x2<=patch['x'][1] and y2<=patch['y'][1] and y2>=patch['y'][0]:
                            centroids['bottom'] = np.vstack((centroids['bottom'],np.array([x1,y1])))
                            centroids['top'] = np.vstack((centroids['top'],np.array([x2,y2])))

                centroids['bottom'] = centroids['bottom'][1:,:]
                centroids['top'] = centroids['top'][1:,:]
                if len(centroids['bottom']) > 0:
                    # Compute mean of centroids
                    for arrow_side in ['bottom','top']:
                        centroids[arrow_side] = np.mean(centroids[arrow_side],axis=0)

                    if centroids['bottom'][1] <= centroids['top'][1]:
                        velocity = (centroids['bottom'][0]-centroids['top'][0], 
                                    centroids['bottom'][1]-centroids['top'][1])
                    else: 
                        velocity = (centroids['top'][0]-centroids['bottom'][0], 
                                    centroids['top'][1]-centroids['bottom'][1])

                    for arrow_side in ['bottom','top']:
                        centroids[arrow_side] = [int(np.round(a)) for a in centroids[arrow_side]]

                    empty_bool = False

                if not empty_bool:
                    # Determine direction and side
                    if velocity[1] < -0.25 and centroids['bottom'][0]>160:
                        X_right = np.vstack((X_right, centroids['bottom']))
                        n_right_side_left_dir += int(velocity[0] < -0.25)
                        n_right_side_right_dir += int(velocity[0] >= -0.25)
                    elif velocity[1] < -0.25 and centroids['bottom'][0]<160:
                        X_left = np.vstack((X_left, centroids['bottom']))
                        n_left_side_left_dir += int(velocity[0] < -0.25)
                        n_left_side_right_dir += int(velocity[0] >= -0.25)

            # Decide lane direction
            if n_right_side_right_dir>=n_right_side_left_dir:
                right_side_dir = ('right', n_right_side_right_dir)
            else:
                right_side_dir = ('left', n_right_side_left_dir)
            if n_left_side_right_dir>=n_left_side_left_dir:
                left_side_dir = ('right', n_left_side_right_dir)
            else:
                left_side_dir = ('left', n_left_side_left_dir)

            if right_side_dir[0] == 'right' and left_side_dir[0] == 'right':
                X_right = np.zeros((1,2))
            if right_side_dir[0] == 'left' and left_side_dir[0] == 'left':
                X_left = np.zeros((1,2))

            X_left = X_left[1:,:]
            X_right = X_right[1:,:]

            # Fit lines if enough points
            if len(X_right)>1:
                print('Computing weighted polynomial interpolation for right lane...')
                right_lane = np.polyfit(X_right[:,0],X_right[:,1],1,w=X_right[:,1])
                y1 = right_lane[0] * 219 + right_lane[1]
                y2 = right_lane[0] * 319 + right_lane[1]
                cv2.line(img_bottom_half_bgr,(219,int(y1)),(319,int(y2)),(150,50,240),5)
                x_start_right = int((25 - right_lane[1])/(right_lane[0]+0.001))
            if len(X_left)>1:
                print('Computing weighted polynomial interpolation for left lane...')
                left_lane = np.polyfit(X_left[:,0],X_left[:,1],1,w=X_left[:,1])
                y1 = left_lane[0] * 0 + left_lane[1]
                y2 = left_lane[0] * 100 + left_lane[1]
                cv2.line(img_bottom_half_bgr,(0,int(y1)),(100,int(y2)),(150,50,240),5)
                x_start_left = int((25 - left_lane[1])/(left_lane[0]+0.001))

            if len(X_right)>1 and len(X_left)>1:
                mid_star = 0.5 * (x_start_right + x_start_left)
                cv2.line(img_bottom_half_bgr,(int(np.clip(mid_star,-10000,10000)),25),(160,100),(0,0,255),5)
            elif len(X_right)>1 and len(X_left)==0:
                mid_star = (25-100)/right_lane[0] + 160
                cv2.line(img_bottom_half_bgr,(int(np.clip(mid_star,-10000,10000)),25),(160,100),(0,0,255),5)
            else: 
                mid_star = 159

            print('Computing steering angle...')
            if np.abs(mid_star-160)<2:
                steering_angle = 90
            else: 
                steering_angle = 90 + np.degrees(np.arctan((mid_star-160)/75.))
                steering_angle = np.clip(steering_angle,55,135)

            if steering_angle is None:
                print("No Present Angle Calculated After processing. Exiting.")
                break
            else:
                stable_steering_angle = stabilize_steering_angle(steering_angle,past_steering_angle)

            # Draw text and steering line
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(stable_steering_angle)
            cv2.putText(img_bottom_half_bgr, text, (110, 30 ), font, 1, (0, 0, 255), 2)

            new_frame = np.concatenate((img_top_half_bgr, img_bottom_half_bgr), axis=0)
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
            cv2.line(new_frame, start_point, end_point, (0, 0, 255), thickness=2)

            cv2.imwrite(os.path.join(path, f"new_frame{getTime()}.jpg"), new_frame)
