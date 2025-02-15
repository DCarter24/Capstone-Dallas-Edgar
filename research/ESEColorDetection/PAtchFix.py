import cv2 # OpenCv Library for image and video processing. 
import numpy as np # Numericla Operations on Arrays. 
import logging # Logging events for debugging.
import datetime # Working with date and time. 
import math # Mathematical Functions. 
import sys # System-Specific parameters and functions. 
import os # Interacting witht the operating system. 


# Constants
# Dimensions for the video capture resolution.
SCREEN_WIDTH = 640  
SCREEN_HEIGHT = 480 
past_steering_angle = 0
row_threshold = 0
path = "/home/pi/repo/Capstone-Dallas-Edgar/research/ESEColorDetection/PatchData"
# Calculate crop height for top 25% of the image
crop_height = int(SCREEN_HEIGHT * 0.25)  # This will be 120 pixels
ifblue = False # Select the HSV bound depending on where we are testing the pi car. 
lower_hsv = 0
upper_hsv = 0

# Initialize the camera for Image capturing. 
camera = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)  
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

#___________________________________________________# 
# Returns the current date and time as a string in the format YYMMDD_HHMMSS. 
def getTime():
    return datetime.datetime.now().strftime("%y%m%d_%H%M%S")

def stabilize_steering_angle(curr_steering_angle, last_steering_angle=None, alpha=0.2):
    # Implementation here for stabilizing the steering angle
    if last_steering_angle is None:
        return int(curr_steering_angle)
    else:
        if 135-last_steering_angle<=5 and curr_steering_angle>= last_steering_angle:
            return np.clip(int(alpha * curr_steering_angle + (1.-alpha) * last_steering_angle),last_steering_angle-1,last_steering_angle+1)
        elif last_steering_angle-55<=5 and curr_steering_angle<= last_steering_angle:
            return np.clip(int(alpha * curr_steering_angle + (1.-alpha) * last_steering_angle),last_steering_angle-1,last_steering_angle+1)
        else:
            return np.clip(int(alpha * curr_steering_angle + (1.-alpha) * last_steering_angle),last_steering_angle-3,last_steering_angle+3)


# Set up time
datestr = getTime(); 

times2Run = {2}


# Main Loop
for i in times2Run:  

    for i in times2Run: 
         successfulRead, raw_image = camera.read() 

         if not successfulRead:
              print("Image not taken successful. ")
              break 
         
         # Rotate the raw image by 180 degrees
         raw_image = cv2.flip(raw_image, -1)

         # row_threshold = 120 # Previous implementation
         row_threshold =  SCREEN_HEIGHT - crop_height  # This will be 360 pixels

         print('Img to color...')
         img_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
         

         print('Cropping top half of the image...')
         #img_bottom_half_bgr = cv2.cvtColor(img_rgb[-row_threshold:,:], cv2.COLOR_RGB2BGR)
         img_bottom_half_bgr = cv2.cvtColor(img_rgb[crop_height:,:], cv2.COLOR_RGB2BGR)
         

         print('Performing HSV color space transformation...')
         #img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
         img_hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)

         print('Creating Top Hald bgr and img crop...')
         #img_top_half_bgr = raw_image[:-row_threshold,:]
         img_top_half_bgr = raw_image[:crop_height,:]
         #img_crop_hsv = img_hsv[-row_threshold:,:]
         img_crop_hsv = img_hsv[crop_height:,:]

         print('Creating binary mask...')
         if ifblue:
              # If True, we are detecting blue lanes for testing. 
              # Define HSV range for blue color
              lower_hsv = np.array([100, 150, 50])
              upper_hsv = np.array([130, 255, 255])
         else:
              # If False, we are detecting white lanes for implementation. 
              # Define HSV range for white color
              lower_hsv = np.array([0, 0, 200])
              upper_hsv = np.array([180, 25, 255])
         
 
         # Create the mask using the new HSV bounds
         mask = cv2.inRange(img_crop_hsv, lower_hsv, upper_hsv)
        
         print('Applying Gaussian blur on mask...')
         mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)  # Gaussian blur

         print('Applying Canny filter...')
         mask_edges = cv2.Canny(mask, 50, 150) # Adjusted to changes made for hsv 

         minLineLength = 12
         maxLineGap = 3
         min_threshold = 5
         
         print('Applying Probabilistic Hough Transform...')
         lines = cv2.HoughLinesP(mask_edges,1,np.pi/180,min_threshold,minLineLength,maxLineGap)
         
         # Save images at various stages
         print("Saving Images without calculating angle.")
         cv2.imwrite(os.path.join(path, f"img_rgb_{getTime()}.jpg"), img_rgb)
         cv2.imwrite(os.path.join(path, f"img_bottom_half_bgr_{getTime()}.jpg"), img_bottom_half_bgr)
         cv2.imwrite(os.path.join(path, f"img_crop_hsv_{getTime()}.jpg"), img_crop_hsv)
         cv2.imwrite(os.path.join(path, f"mask_{getTime()}.jpg"), mask)
         cv2.imwrite(os.path.join(path, f"mask_blurred_{getTime()}.jpg"), mask_blurred)
         cv2.imwrite(os.path.join(path, f"mask_edges_{getTime()}.jpg"), mask_edges)

         # Adjusted 'list_patch' with corrected 'y' values
         original_list_patch = [
              # First set of patches (original y: (120, 100) adjusted)
              {'x': (0,25),     'y': (240 - crop_height, 180 - crop_height)},
              {'x': (25,50),    'y': (240 - crop_height, 180 - crop_height)},
              {'x': (50,75),    'y': (240 - crop_height, 180 - crop_height)},
              {'x': (75,100),   'y': (240 - crop_height, 180 - crop_height)},
              {'x': (100,125),  'y': (240 - crop_height, 180 - crop_height)},
              {'x': (194,219),  'y': (240 - crop_height, 180 - crop_height)},
              {'x': (219,244),  'y': (240 - crop_height, 180 - crop_height)},
              {'x': (244,269),  'y': (240 - crop_height, 180 - crop_height)},
              {'x': (269,294),  'y': (240 - crop_height, 180 - crop_height)},
              {'x': (294,319),  'y': (240 - crop_height, 180 - crop_height)},

              # Second set of patches (original y: (100, 75) adjusted)
              {'x': (0,25),     'y': (180 - crop_height, 105 - crop_height)},
              {'x': (25,50),    'y': (180 - crop_height, 105 - crop_height)},
              {'x': (50,75),    'y': (180 - crop_height, 105 - crop_height)},
              {'x': (75,100),   'y': (180 - crop_height, 105 - crop_height)},
              {'x': (100,125),  'y': (180 - crop_height, 105 - crop_height)},
              {'x': (194,219),  'y': (180 - crop_height, 105 - crop_height)},
              {'x': (219,244),  'y': (180 - crop_height, 105 - crop_height)},
              {'x': (244,269),  'y': (180 - crop_height, 105 - crop_height)},
              {'x': (269,294),  'y': (180 - crop_height, 105 - crop_height)},
              {'x': (294,319),  'y': (180 - crop_height, 105 - crop_height)},

              # Third set of patches (original y: (75, 50) adjusted)
              {'x': (0,25),     'y': (105 - crop_height, 30 - crop_height)},
              {'x': (25,50),    'y': (105 - crop_height, 30 - crop_height)},
              {'x': (50,75),    'y': (105 - crop_height, 30 - crop_height)},
              {'x': (75,100),   'y': (105 - crop_height, 30 - crop_height)},
              {'x': (100,125),  'y': (105 - crop_height, 30 - crop_height)},
              {'x': (194,219),  'y': (105 - crop_height, 30 - crop_height)},
              {'x': (219,244),  'y': (105 - crop_height, 30 - crop_height)},
              {'x': (244,269),  'y': (105 - crop_height, 30 - crop_height)},
              {'x': (269,294),  'y': (105 - crop_height, 30 - crop_height)},
              {'x': (294,319),  'y': (105 - crop_height, 30 - crop_height)},

              # Fourth set of patches (original y: (50, 25) adjusted)
              {'x': (75,100),   'y': (30 - crop_height, 0)},
              {'x': (100,125),  'y': (30 - crop_height, 0)},
              {'x': (194,219),  'y': (30 - crop_height, 0)},
              {'x': (219,244),  'y': (30 - crop_height, 0)},

              # Fifth set of patches (original y: (25, 0) adjusted)
              # Adjusted to (0, 0) since negative values are outside the image
              {'x': (100,125),  'y': (0, 0)},
              {'x': (194,219),  'y': (0, 0)},
          ]
         # Adjust scaling factors
         x_scale = SCREEN_WIDTH / 320  # 640 / 320 = 2
         y_scale = (SCREEN_HEIGHT - crop_height) / 120  # 360 / 120 = 3
         list_patch = []

         # Optionally, evaluate the adjusted 'y' values now to get integer values
         # Evaluate the adjusted 'y' values
         for patch in original_list_patch:
              # Calculate adjusted 'y' values
              y0 = int(patch['y'][0] * y_scale)
              y1 = int(patch['y'][1] * y_scale)
              # Ensure 'y' coordinates are ordered from top to bottom
              if y0 > y1:
                   y0, y1 = y1, y0
               
              # Similarly ensure 'x' coordinates are integers
              x0 = int(patch['x'][0] * x_scale)
              x1 = int(patch['x'][1] * x_scale)

              list_patch.append({'x': (x0, x1), 'y': (y0, y1)})
              


         for idx, patch in enumerate(list_patch):
               x0, x1 = patch['x']
               y0, y1 = patch['y']
               
               # Debugging: Print patch coordinates
               print(f"Patch {idx}: x0={x0}, x1={x1}, y0={y0}, y1={y1}")
               
               # Draw the patch
               cv2.line(img_bottom_half_bgr, (x0, y0), (x1, y0), (0,165,255), 1)  # Top line
               cv2.line(img_bottom_half_bgr, (x0, y1), (x1, y1), (0,165,255), 1)  # Bottom line
               cv2.line(img_bottom_half_bgr, (x0, y0), (x0, y1), (0,165,255), 1)  # Left line
               cv2.line(img_bottom_half_bgr, (x1, y0), (x1, y1), (0,165,255), 1)  # Right line

         
         print("Saving Image With Lines.")
         cv2.imwrite(os.path.join(path, f"image_lines{getTime()}.jpg"), img_bottom_half_bgr)
         
         if lines is None: 
             print("No Lines Detected. Exiting Loop")
             break
         else:
              X_left=np.zeros((1,2))
              X_right=np.zeros((1,2))
              n_right_side_right_dir = 0
              n_right_side_left_dir = 0
              n_left_side_right_dir = 0
              n_left_side_left_dir = 0

              print('Computing centroids and mean direction...') 
              for patch in list_patch: 
                   centroids = {'bottom': np.zeros((1,2)),'top': np.zeros((1,2))}
                   velocity = (None,None)
                   

                   for line in lines:
                        x1,y1,x2,y2 = line[0]
                        if x1>=patch['x'][0] and x1<=patch['x'][1] and y1<=patch['y'][0] and y1>=patch['y'][1]:
                             if x2>=patch['x'][0] and x2<=patch['x'][1] and y2<=patch['y'][0] and y2>=patch['y'][1]:
                                  centroids['bottom'] = np.vstack((centroids['bottom'],np.array([x1,y1])))
                                  centroids['top'] = np.vstack((centroids['top'],np.array([x2,y2])))

                   centroids['bottom'] = centroids['bottom'][1:,:]
                   centroids['top'] = centroids['top'][1:,:]
                   if len(centroids['bottom'])>0:
                        for arrow_side in ['bottom','top']:
                            centroids[arrow_side] = np.mean(centroids[arrow_side],axis=0)

                        if centroids['bottom'][1] <= centroids['top'][1]:
                            velocity = (centroids['bottom'][0]-centroids['top'][0], centroids['bottom'][1]-centroids['top'][1])
                        else: 
                             velocity = (centroids['top'][0]-centroids['bottom'][0], centroids['top'][1]-centroids['bottom'][1])

                        for arrow_side in ['bottom','top']:
                             centroids[arrow_side] = [int(np.round(a)) for a in centroids[arrow_side]]
                        
                        empty_bool = False
                   else:
                    empty_bool = True

                    if not empty_bool:
                         if velocity[1] < -0.25 and centroids['bottom'][0]>160: #velocity[0] < -0.25 and 
                            X_right = np.vstack((X_right, centroids['bottom']))
                            n_right_side_left_dir += int(velocity[0] < -0.25)
                            n_right_side_right_dir += int(velocity[0] >= -0.25)
                         elif velocity[1] < -0.25 and centroids['bottom'][0]<160: #velocity[0] > 0.25 and 
                            X_left = np.vstack((X_left, centroids['bottom']))
                            n_left_side_left_dir += int(velocity[0] < -0.25)
                            n_left_side_right_dir += int(velocity[0] >= -0.25)

              if n_right_side_right_dir>=n_right_side_left_dir:
                   right_side_dir = ('right', n_right_side_right_dir)
              else:
                   right_side_dir = ('left', n_right_side_left_dir)
              if n_left_side_right_dir>=n_left_side_left_dir:
                     left_side_dir = ('right', n_left_side_right_dir)
              else:
                   left_side_dir = ('left', n_left_side_left_dir)
              if right_side_dir[0] == 'right' and left_side_dir[0] == 'right':
                    X_right=np.zeros((1,2))

              if right_side_dir[0] == 'left' and left_side_dir[0] == 'left':
                    X_left=np.zeros((1,2))

              X_left = X_left[1:,:]
              X_right = X_right[1:,:]

              if len(X_right)>1:
                   print('Computing weighted polynomial interpolation...')
                   right_lane = np.polyfit(X_right[:,0],X_right[:,1],1,w=X_right[:,1])
                   y1 = right_lane[0] * 219 + right_lane[1]
                   y2 = right_lane[0] * 319 + right_lane[1]
                   cv2.line(img_bottom_half_bgr,(219,int(y1)),(319,int(y2)),(150,50,240),5) #150,50,255

                   x_start_right = int((25 - right_lane[1])/(right_lane[0]+0.001))
              if len(X_left)>1:
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
                   steering_angle = np.clip(steering_angle,55, 135) # Limit value to within range

              # Stabalize Steering angle
              if steering_angle is None: 
                   print("No Present Angle Calculated After processing. Exiting. ")
                   break
              else:
                   stable_steering_angle = stabilize_steering_angle(steering_angle,past_steering_angle)

              # setup text
              font = cv2.FONT_HERSHEY_SIMPLEX
              text = str(stable_steering_angle)
              # get boundary of this text
              textsize = cv2.getTextSize(text, font, 1, 2)[0]
              # get coords based on boundary
              textX = 110
              textY = 30
              # add text centered on image
              cv2.putText(img_bottom_half_bgr, text, (textX, textY ), font, 1, (0, 0, 255), 2)

              new_frame = np.concatenate((img_top_half_bgr, img_bottom_half_bgr), axis=0)

              # Draw steering angle line
              height, width, _ = new_frame.shape

              # Starting point: bottom center of the image
              start_point = (int(width / 2), int(height))

              # Calculate angle from vertical in radians
              angle_from_vertical = stable_steering_angle - 90
              angle_rad = np.radians(angle_from_vertical)

              # Line length in pixels
              line_length = 100  # Adjust the length as needed

              # Calculate end point
              end_point_x = int(start_point[0] + line_length * np.sin(angle_rad))
              end_point_y = int(start_point[1] - line_length * np.cos(angle_rad))

              # Ensure end point is within image boundaries
              end_point_x = max(0, min(end_point_x, width - 1))
              end_point_y = max(0, min(end_point_y, height - 1))

              end_point = (end_point_x, end_point_y)

              # Draw the red line representing the steering angle
              cv2.line(new_frame, start_point, end_point, (0, 0, 255), thickness=2)

              # Save new image: 
              cv2.imwrite(os.path.join(path, f"new_frame{getTime()}.jpg"), new_frame)

   
