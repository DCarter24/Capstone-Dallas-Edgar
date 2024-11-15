import cv2

filename  = input("Enter name of file to process: ")
orig_img  = cv2.imread(filename)

height, width = orig_img.shape[:2]

# might want to resize some images, but used the original in this program
scale = 0.2
new_size  = (int(height*scale), int(width*scale))
small_img = cv2.resize(orig_img, new_size, interpolation=cv2.INTER_LINEAR) 

# Converts the orginal image ot grayscale (gray) and HSV color space (hsv_img). 
gray      = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
hsv_img   = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
new_img   = orig_img.copy()

def blank(x):       # null function for trackbar
   pass;

# Prepares labels fo rthe vertical and hosizontal trachbars. 
vertstr = (f'vert 0: {height-1}')
horzstr = (f'hort 0: {width-1}')

# Creates a window called 'window' 
cv2.namedWindow('window', cv2.WINDOW_NORMAL)
# Resizes Window to match the image dimensions. 
cv2.resizeWindow('window', width, height)
# Adds two trackbars ot the window for selecting vertical and horizontal coordinates. 
cv2.createTrackbar(vertstr, 'window', 0, height-1, blank)
cv2.createTrackbar(horzstr, 'window', 0, width-1,  blank)

# Main Loop. 
while True:
   # REtrieves the current positions of the vertical (lc) and horizontal (hc) trackbars. 
   lc = cv2.getTrackbarPos(vertstr, 'window')
   hc = cv2.getTrackbarPos(horzstr, 'window')

   # Draws a red circle at the selected coordinates on new_img. 
   new_img = cv2.circle(new_img, (hc, lc), radius=int(width*0.005), 
                        color=(0,0,255), thickness=int(width*0.002))
  
   # Retrieves the HSV values at the selected point. 
   hval = hsv_img[lc, hc, 0]   
   sval = hsv_img[lc, hc, 1]
   vval = hsv_img[lc, hc, 2]

   # Displays the updated image with the circle. 
   cv2.imshow('window', new_img)
   outstring = (f'H: {hval}\tS: {sval}\tV: {vval}')     # prints on top line
   # Updates the window title to show the HSV values. 
   cv2.setWindowTitle('window', outstring)
   
   # Checks if the space bar (ASCII code 32) is pressed to exit the loop. 
   if cv2.waitKey(1) == 32:     # stop when space bar hit
      cv2.destroyAllWindows()
      outstring = (f'({lc}, {hc})  H:{hval} S:{sval} V:{vval}')
      # put circle on final spot, 
      # draws a red circle on the original image at the selected point. 
      orig_img = cv2.circle(orig_img, (hc, lc), radius=int(width*0.005), 
                            color=(0,0,255), thickness=int(width*0.002))
      # put the values in the upper left
      # Writes the coordinates and HSV values onto the image. 
      orig_img = cv2.putText(orig_img, outstring, (height//10, width//10), 
                             cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
      
      # Saves the modified image as 'new_img.jpg'. 
      cv2.imwrite("new_img.jpg", orig_img)
      break;
