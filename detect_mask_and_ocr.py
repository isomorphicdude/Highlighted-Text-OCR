import cv2
import numpy as np
import torch
import easyocr

# Reading the image

img = cv2.imread('image.jpg')

#define kernel size  
kernel = np.ones((7,7),np.uint8)


# convert to hsv colorspace 
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower bound and upper bound for Green color 
# lower_bound = np.array([50, 20, 20])     
# upper_bound = np.array([100, 255, 255])

# lower bound and upper bound for Yellow color 
lower_bound = np.array([20, 80, 80])     
upper_bound = np.array([30, 255, 255])

# find the colors within the boundaries
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Remove unnecessary noise from mask
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Segment only the detected region
segmented_img = cv2.bitwise_and(img, img, mask=mask)

output = cv2.resize(segmented_img, (960, 540))

cv2.imwrite('modified',output)

reader = easyocr.Reader(['de', 'en'])

result = reader.readtext('modified.jpg')