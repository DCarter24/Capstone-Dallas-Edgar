import matplotlib.pyplot as plt
import numpy as np
import math as m
import argparse
import time
import cv2


start_time = time.time()
cur_time = start_time
capture_time = start_time

# Main Part of program
while (start_time + args.tim > cur_time):
	cur_time = time.time()
	#print(f'cur_time: {cur_time}')
	
	if (capture_time + args.delay < cur_time):
		print('___________________________________________________')
		capture_time = cur_time #- start_time
		real_cap_baby = cur_time - start_time
		
		print(f'capture_time:{real_cap_baby} ')
		
		img = car.get_image() 
		
		#PWM1 = NPWM(img)
		
		# new_Angle = find_angle(img)
		
		#___________________Find Angle Function__________________________#
		print("__")
		print("Inside Find Angle")
		# Basic Printing of Characteristics of the Image 
		print(f'Image Size: {img.size}') 
		print(f'Image Shape: {img.shape}')
		
		# convert img from RGB to HSV 
		hsv = cv2.cvtColor(img,  cv2.COLOR_BGR2HSV)
		
		# 1st set of values  form lower limits, 
		# the second the upper using variables 
		
		lower_level = (95, 50, 50)
		upper_level = (125,255, 255)
		
		# Creating a mask 
		mask = cv2.inRange(hsv, lower_level , upper_level)
		
		
		#Locating the Region of a specific color by finding the moment
		M = cv2.moments(mask)  
		if(M["m00"] <= 0.00001):
			print("NO IMAGE FOUND")
			new_Angle = 360 
		else:
			print("IMAGE FOUND BITCHES")
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
		
			# Creating a blur image 
			mask_blur = cv2.blur(mask,(5,5))
			
			# creating the threshold 
			thresh = cv2.threshold(mask_blur, 200, 255, cv2.THRESH_BINARY)[1]
			
			# Make red dot at the center of the specified color
			img_with_blue_circle = cv2.circle(img, (cX, cY),  5, (0,0,255), 2) 
			cv2.imwrite('testing.png',img_with_blue_circle )
			# calculate the angle of the red dot using imported function 
			
			#angle = origin_calcualtion(img_with_blue_circle, cY, cX) # Call function
			
			
			#___________________Origin Calculation_________________________#
			print("__")
			print("Inside Origin Calculation")
			# Calculate the variables needed for the equation. 
			
			# Get the original dimensions of the image
			height_total, width_total, _ = img.shape
			
			# divide height by two to get the center of the image, the reference point
			half_w = (width_total/2) 
			
			# calculate the opposite side of the angle
			adjacent_side = height_total - cY 
			
			# calculate the adjacent side
			opposite_side =  cX -   half_w 
			
			# calculate angle
			theta = m.atan(opposite_side/adjacent_side) 
			theta= theta*(180/m.pi)
			print('theta ' + str(theta))
			
			print( 'pixel x '+str(cX) + ' pixely ' +str(cY))
			
			if(args.debug):
				# Save the mask created as an image
				cv2.imwrite('mask.jpg', mask)
				# Save the mask_blur
				cv2.imwrite('img_blur.jpg', mask_blur)
				# Save image with the filter
				cv2.imwrite('thresh.jpg', thresh)
				# Save image with the red dot 
				cv2.imwrite('blue_circle.jpg', img_with_blue_circle) 
				 
				# Printing angle 
				#print(f'Angle calculated: {angle}')
				# printing the center 
				print (f"Center: ({cX} , {cY})")
				 
				#cv2.imshow('mask.jpg')
				#cv2.imshow('img_blur.jpg')
				#cv2.imshot('thresh.jpg')
				#cv2.imshow('blue_circle.jpg')

			print('Made img with circle at the center of the specified color') 
		
		if(M["m00"]  <= 0.00001):
			print("SECOND CHECK, ANGLE IS 360 :(")
			new_angle = 360 
		elif(not (M["m00"] == 0) ):
			print("New Angle MADE")
			new_Angle = theta 
		
		#___________________NPWM Function______________________________#
		print("__")
		print("Inside NPWM ")
		
		if(not (new_Angle  == 360) or new_Angle < 360): 
			PWM = PWM - delta * scale_factor * new_Angle
			print('new pwm '+ str(PWM) )
			print( 'new_Angle ' + str(new_Angle) )
		if (PWM < lower):
			PWM = lower
			print("PWM Changed to Lower: " + str(PWM))
		elif(PWM > upper):
			PWM = upper
			print("PWM Changed to upper: " + str(PWM))
		
		# Change Direction of View
		print(f'FINAL PWM: {PWM}')
		car.set_swivel_servo(PWM)
		
		
		#time.sleep(.5)




car.stop()
print("End" )
