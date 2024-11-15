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

# Initialize the camera for Image capturing. 
camera = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)  
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

#___________________________________________________# 
# Returns the current date and time as a string in the format YYMMDD_HHMMSS. 
def getTime():
    return datetime.datetime.now().strftime("%y%m%d_%H%M%S")


# Set up time
datestr = getTime(); 