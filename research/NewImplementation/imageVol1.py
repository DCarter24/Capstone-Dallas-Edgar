import cv2 # OpenCv Library for image and video processing. 
import numpy as np # Numericla Operations on Arrays. 
import logging # Logging events for debugging.
import datetime # Working with date and time. 
import math # Mathematical Functions. 
import sys # System-Specific parameters and functions. 
import os # Interacting witht the operating system. 

# Constants
_INITIAL_SPEED = 0 # Initial speed of the vehicle. 
_SCREEN_WIDTH = 640  # Dimensions for the video capture resolution. 
_SCREEN_HEIGHT = 480 
_SHOW_IMAGE = False  # Flag ot determine whether to display images during processing. 

# Define base path for saving videos and images
base_video_path = "/home/pi/repo/Capstone-Dallas-Edgar/research/NewImplementation/data/videos"
base_media_path = "/home/pi/repo/Capstone-Dallas-Edgar/research/NewImplementation/data/images"


# Returns the current date and time as a string in the format YYMMDD_HHMMSS. 
def getTime():
    return datetime.datetime.now().strftime("%y%m%d_%H%M%S")


# Initialize the camera for video capture. 
def setup_camera():
    """Set up the camera and return it"""
    logging.debug('Setting up camera')
    camera = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)  
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, _SCREEN_WIDTH)  
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, _SCREEN_HEIGHT)
    return camera # Returns object for capturing video formats.


# Creates a object to record videos. 
def create_video_recorder(base_path, filename, fourcc):
    """Create directory if not exists and return a video recorder"""
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    full_path = os.path.join(base_path, filename)
    return cv2.VideoWriter(full_path, fourcc, 20.0, (_SCREEN_WIDTH, _SCREEN_HEIGHT))

# Releases resources and perfomrs clean up when script is stopped. 
def cleanup(camera, video_orig, video_lane):
    """Cleanup resources"""
    logging.info('Stopping the car, resetting hardware.')
    camera.release()
    video_orig.release()
    video_lane.release()
    cv2.destroyAllWindows()

# Draws rectangular patch (regions of interest) on the image. 
def draw_patches(img,patch_):
    # Implementation here for drawing patches
    '''
    cv2.line(): Method used to draw a line on any image: 
        Usage: 
            cv2.line(image, start_point, end_point, color, thickness). 

    patch_: A dictionary containing the coordinates of the patch: 
            patch_['x']: Tuple with x-coordinates (x_start, x_end). 
            patch_['y']: Tuple with y-coordinates (y_start, y_end). 
    Color: 
        (0, 165, 255): corresponds to orange in BGR Format. 
    '''

    # Draws four sides of a rectangle based on the coordinates partc_. 
    cv2.line(img,(patch_['x'][0],patch_['y'][0]),(patch_['x'][1],patch_['y'][0]),(0,165,255),1) # First line. 
    cv2.line(img,(patch_['x'][0],patch_['y'][1]),(patch_['x'][1],patch_['y'][1]),(0,165,255),1) # Second line. 
    cv2.line(img,(patch_['x'][0],patch_['y'][0]),(patch_['x'][0],patch_['y'][1]),(0,165,255),1) # Third line. 
    cv2.line(img,(patch_['x'][1],patch_['y'][0]),(patch_['x'][1],patch_['y'][1]),(0,165,255),1) # Fourth line. 
    return img # returns the image on which to draw. 

# Computers the centroids and velocities of line segments within a given patch. 
def get_centroids_from_patches(list_lines,patch_):
    # Implementation here for getting centroids from patches

    ''' 
    
    
    '''
    centroids = {'bottom': np.zeros((1,2)),'top': np.zeros((1,2))}
    velocity = (None,None)

    for line in list_lines:
        x1,y1,x2,y2 = line[0]
        if x1>=patch_['x'][0] and x1<=patch_['x'][1] and y1<=patch_['y'][0] and y1>=patch_['y'][1]:
            if x2>=patch_['x'][0] and x2<=patch_['x'][1] and y2<=patch_['y'][0] and y2>=patch_['y'][1]:
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

    return centroids, velocity, empty_bool


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


def pipeline_lane_detector(frame_, image_path, past_steering_angle=None):

    row_threshold = 120

    print('Img to color...')
    img_rgb = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)

    print('Cropping top half of the image...')
    img_bottom_half_bgr = cv2.cvtColor(img_rgb[-row_threshold:,:], cv2.COLOR_RGB2BGR)


    print('Performing HSV color space transformation...')
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    print('Creating Top Hald bgr and img crop...')
    img_top_half_bgr = frame_[:-row_threshold,:]
    img_crop_hsv = img_hsv[-row_threshold:,:]


    print('Creating binary mask...')
    bound = (np.array([0, 0, 0]), np.array([255, 255, 50]))
    mask = cv2.inRange(img_crop_hsv, bound[0], bound[1])
    
    print('Applying Gaussian blur on mask...')
    mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)  # Gaussian blur

    #mask_blurred = cv2.blur(mask,(5,5))

    
    print('Applying Canny filter...')
    mask_edges = cv2.Canny(mask, 200, 400)




    minLineLength = 12
    maxLineGap = 3
    min_threshold = 5

    print('Applying Probabilistic Hough Transform...')
    lines = cv2.HoughLinesP(mask_edges,1,np.pi/180,min_threshold,minLineLength,maxLineGap)

    # Save images at various stages
    cv2.imwrite(os.path.join(image_path, f"img_rgb_{getTime()}.jpg"), img_rgb)
    cv2.imwrite(os.path.join(image_path, f"img_bottom_half_bgr_{getTime()}.jpg"), img_bottom_half_bgr)
    cv2.imwrite(os.path.join(image_path, f"img_crop_hsv_{getTime()}.jpg"), img_crop_hsv)
    cv2.imwrite(os.path.join(image_path, f"mask_{getTime()}.jpg"), mask)
    cv2.imwrite(os.path.join(image_path, f"mask_blurred_{getTime()}.jpg"), mask_blurred)
    cv2.imwrite(os.path.join(image_path, f"mask_edges_{getTime()}.jpg"), mask_edges)


    list_patch = [{'x': (0,25),'y': (120,100)}, {'x': (25,50),'y': (120,100)}, {'x': (50,75),'y': (120,100)},{'x': (75,100),'y': (120,100)}, {'x': (100,125),'y': (120,100)}]
    list_patch += [{'x': (194,219),'y': (120,100)},{'x': (219,244),'y': (120,100)}, {'x': (244,269),'y': (120,100)}, {'x': (269,294),'y': (120,100)},{'x': (294,319),'y': (120,100)}]

    list_patch += [{'x': (0,25),'y': (100,75)}, {'x': (25,50),'y': (100,75)}, {'x': (50,75),'y': (100,75)},{'x': (75,100),'y': (100,75)}, {'x': (100,125),'y': (100,75)}]
    list_patch += [{'x': (194,219),'y': (100,75)},{'x': (219,244),'y': (100,75)}, {'x': (244,269),'y': (100,75)}, {'x': (269,294),'y': (100,75)},{'x': (294,319),'y': (100,75)}]

    list_patch += [{'x': (0,25),'y': (75,50)}, {'x': (25,50),'y': (75,50)}, {'x': (50,75),'y': (75,50)},{'x': (75,100),'y': (75,50)}, {'x': (100,125),'y': (75,50)}]
    list_patch += [{'x': (194,219),'y': (75,50)},{'x': (219,244),'y': (75,50)}, {'x': (244,269),'y': (75,50)}, {'x': (269,294),'y': (75,50)},{'x': (294,319),'y': (75,50)}]

    list_patch += [{'x': (75,100),'y': (50,25)}, {'x': (100,125),'y': (50,25)}]
    list_patch += [{'x': (194,219),'y': (50,25)},{'x': (219,244),'y': (50,25)}]

    list_patch += [{'x': (100,125),'y': (25,0)}]
    list_patch += [{'x': (194,219),'y': (25,0)}]

    for patch in list_patch:
        img_bottom_half_bgr = draw_patches(img_bottom_half_bgr, patch)

    if lines is None:
        return frame_, past_steering_angle
    else:
        X_left=np.zeros((1,2))
        X_right=np.zeros((1,2))
        n_right_side_right_dir = 0
        n_right_side_left_dir = 0
        n_left_side_right_dir = 0
        n_left_side_left_dir = 0
        print('Computing centroids and mean direction...')        
        for patch in list_patch:
            centroids, velocity, empty_bool = get_centroids_from_patches(lines, patch)
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
            #mid_star = 25/right_lane[0] - 160 - right_lane[1]/right_lane[0]
            mid_star = (25-100)/right_lane[0] + 160
            cv2.line(img_bottom_half_bgr,(int(np.clip(mid_star,-10000,10000)),25),(160,100),(0,0,255),5)
        elif len(X_right)==0 and len(X_left)>1:
            mid_star = (25-100)/left_lane[0] + 160
            cv2.line(img_bottom_half_bgr,(int(np.clip(mid_star,-10000,10000)),25),(160,100),(0,0,255),5)
        else:
            mid_star = 159

        print('Computing steering angle...')
        if np.abs(mid_star-160)<2:
            steering_angle = 90
        else:
            steering_angle = 90 + np.degrees(np.arctan((mid_star-160)/75.))
            steering_angle = np.clip(steering_angle,55, 135)

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
        return new_frame, stable_steering_angle


def follow_lane(frame, image_path, current_steering_angle=90):
    """Process the frame to follow the lane"""
    frame_lanes, new_steering_angle = pipeline_lane_detector(frame, image_path, current_steering_angle) 
    return frame, new_steering_angle

def drive(camera, video_orig, video_lane):
    """Drive the car and follow lanes"""
    logging.info('Starting to drive at speed %s...' % _INITIAL_SPEED)
    current_steering_angle = 90  # Initial steering angle
    while camera.isOpened():
        ret, image_lane = camera.read()
        if not ret:
            break
        image_path = os.path.join(base_media_path, f"frame_{getTime()}.jpg")
        image_lane, current_steering_angle = follow_lane(image_lane, image_path,current_steering_angle )
        #video_orig.write(image_lane)
        #video_lane.write(image_lane)
        if _SHOW_IMAGE:
            cv2.imshow('Lane Lines', image_lane)
        '''
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        '''
def show_image(title, frame, show=_SHOW_IMAGE):
    """Display an image if _SHOW_IMAGE is True"""
    if show:
        cv2.imshow(title, frame)

# Setup
logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s:%(asctime)s: %(message)s')
logging.info('Setting up the system...')



camera = setup_camera()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
datestr = getTime(); 
video_orig = create_video_recorder(base_video_path, f"Original_car_video_{datestr}.avi",fourcc)
video_lane = create_video_recorder(base_video_path, f"lane_car_video_{datestr}.avi", fourcc)

# Main driving loop
try:
    drive(camera, video_orig, video_lane)
finally:
    cleanup(camera, video_orig, video_lane)
