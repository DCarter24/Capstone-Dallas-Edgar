import cv2

# open camera
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)

# set dimensions
 # original code 
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
# Changing it too 1920x1080 (Full HD)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# take frame
ret, frame = cap.read()
# write frame to file
cv2.imwrite('image.jpg', frame)
# release camera
cap.release()
