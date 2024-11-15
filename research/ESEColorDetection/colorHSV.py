import cv2

filename  = input("Enter name of file to process: ")
orig_img  = cv2.imread(filename)

height, width = orig_img.shape[:2]

# might want to resize some images, but used the original in this program
scale = 0.2
new_size  = (int(height*scale), int(width*scale))
small_img = cv2.resize(orig_img, new_size, interpolation=cv2.INTER_LINEAR) 

gray      = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
hsv_img   = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
new_img   = orig_img.copy()

def blank(x):       # null function for trackbar
   pass;

vertstr = (f'vert 0: {height-1}')
horzstr = (f'hort 0: {width-1}')
cv2.namedWindow('window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('window', width, height)
cv2.createTrackbar(vertstr, 'window', 0, height-1, blank)
cv2.createTrackbar(horzstr, 'window', 0, width-1,  blank)

while True:

   lc = cv2.getTrackbarPos(vertstr, 'window')
   hc = cv2.getTrackbarPos(horzstr, 'window')

   new_img = cv2.circle(new_img, (hc, lc), radius=int(width*0.005), 
                        color=(0,0,255), thickness=int(width*0.002))
  
   hval = hsv_img[lc, hc, 0]   
   sval = hsv_img[lc, hc, 1]
   vval = hsv_img[lc, hc, 2]

   cv2.imshow('window', new_img)
   outstring = (f'H: {hval}\tS: {sval}\tV: {vval}')     # prints on top line
   cv2.setWindowTitle('window', outstring)
   
   if cv2.waitKey(1) == 32:     # stop when space bar hit
      cv2.destroyAllWindows()
      outstring = (f'({lc}, {hc})  H:{hval} S:{sval} V:{vval}')
      # put circle on final spot
      orig_img = cv2.circle(orig_img, (hc, lc), radius=int(width*0.005), 
                            color=(0,0,255), thickness=int(width*0.002))
      # put the values in the upper left
      orig_img = cv2.putText(orig_img, outstring, (height//10, width//10), 
                             cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
      cv2.imwrite("new_img.jpg", orig_img)
      break;
