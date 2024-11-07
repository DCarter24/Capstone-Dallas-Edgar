import logging
import picar
import cv2
import datetime
from my_hand_coded_lane_follower import MyHandCodedLaneFollower

# Constants
_INITIAL_SPEED = 0
_SCREEN_WIDTH = 640  
_SCREEN_HEIGHT = 480  
_SHOW_IMAGE = True

def setup_camera():
    """Set up the camera and return it"""
    logging.debug('Setting up camera')
    # Use the specific device path and V4L for compatibility 
    camera = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)  
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, _SCREEN_WIDTH)  # Set resolution
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, _SCREEN_HEIGHT)
    return camera

def create_video_recorder(path, fourcc):
    """Create and return a video recorder"""
    return cv2.VideoWriter(path, fourcc, 20.0, (_SCREEN_WIDTH, _SCREEN_HEIGHT))

def cleanup(camera, video_orig, video_lane):
    """Cleanup resources"""
    logging.info('Stopping the car, resetting hardware.')
    camera.release()
    video_orig.release()
    video_lane.release()
    cv2.destroyAllWindows()

def drive(camera, lane_follower, video_orig, video_lane):
    """Drive the car and follow lanes"""
    logging.info('Starting to drive at speed %s...' % _INITIAL_SPEED)
    while camera.isOpened():
        ret, image_lane = camera.read()
        if not ret:
            break
        image_objs = image_lane.copy()
        video_orig.write(image_lane)

        image_lane = lane_follower.follow_lane(image_lane)
        video_lane.write(image_lane)
        show_image('Lane Lines', image_lane)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def show_image(title, frame, show=_SHOW_IMAGE):
    """Display an image if _SHOW_IMAGE is True"""
    if show:
        cv2.imshow(title, frame)

# Setup
logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s:%(asctime)s: %(message)s')
picar.setup()
logging.info('Creating a DeepPiCar...')

# Initialize components
camera = setup_camera()
lane_follower = MyHandCodedLaneFollower(camera)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
datestr = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
video_orig = create_video_recorder('../data/tmp/car_video%s.avi' % datestr, fourcc)
video_lane = create_video_recorder('../data/tmp/car_video_lane%s.avi' % datestr, fourcc)

# Main driving loop
try:
    drive(camera, lane_follower, video_orig, video_lane)
finally:
    cleanup(camera, video_orig, video_lane)
