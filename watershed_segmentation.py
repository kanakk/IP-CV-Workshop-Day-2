# -*- coding: utf-8 -*-
"""
Author - Kanak Kawadiwale on Thu Nov  5 20:03:43 2020

@author: ag
"""

#importing the required libraries
import cv2
from matplotlib import pyplot as plt
import numpy as np

#reading the static image
img = cv2.imread('images/coin.jpg')

#spliting the Image colorspace
b,g,r = cv2.split(img)

#Combining the colorspace
rgb_img = cv2.merge([r,g,b])

#Converting the image into Grayscale image for processing 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Applying OTSU thresholding to the clear the noise within image
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# 2*2 Kernel for removing the noise 
kernel = np.ones((2,2),np.uint8)

#Applying morphological operation - CLOSING
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

# Finding the dominant bakground in image
sure_bg = cv2.dilate(closing,kernel,iterations=3)

# Finding the dominant foreground in image using the depth analysis
#Parameters - Source, Distance Type, Mask Size 
dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)


# Thresholding the image to get the dominant foreground - Coins
ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

# Find the unnecesary region in the processed image and remove it to get the clear image
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg,sure_fg)

# Fetch the markers in the images to understand the connected boundaries within
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Eliminate the unncessary part by allocating 0 to its pixels
markers[unknown==255] = 0

#Finallly Apply the watershed segmentation algorithm to the 
#necessary part to detect the boundaries
markers = cv2.watershed(img,markers)

#Now to make this markers visible, give it a color to be seen (in this case it is green)
img[markers == -1] = [0,255,0]


#Now Plotting this images to compare the result and display
plt.subplot(421),plt.imshow(rgb_img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(422),plt.imshow(thresh, 'gray')
plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])

plt.subplot(423),plt.imshow(closing, 'gray')
plt.title("morphologyEx:Closing:2x2"), plt.xticks([]), plt.yticks([])
plt.subplot(424),plt.imshow(sure_bg, 'gray')
plt.title("Dilation"), plt.xticks([]), plt.yticks([])

plt.subplot(425),plt.imshow(dist_transform, 'gray')
plt.title("Distance Transform"), plt.xticks([]), plt.yticks([])
plt.subplot(426),plt.imshow(sure_fg, 'gray')
plt.title("Thresholding"), plt.xticks([]), plt.yticks([])

plt.subplot(427),plt.imshow(unknown, 'gray')
plt.title("Unknown"), plt.xticks([]), plt.yticks([])

plt.subplot(428),plt.imshow(img, 'gray')
plt.title("Result from Watershed"), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

cv2.imshow('Watershed image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()