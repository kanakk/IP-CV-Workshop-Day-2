# -*- coding: utf-8 -*-
"""
Author - Kanak Kawadiwale on Thu Nov  5 20:03:43 2020

@author: ag
"""

#importing the required libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#Read the two images that needs to be matched with to match features
original_image = cv.imread('images/feature1.jpg',cv.IMREAD_GRAYSCALE)

cropped_image = cv.imread('images/feature2.jpg',cv.IMREAD_GRAYSCALE)

cv.imshow('Watershed image',original_image)
cv.waitKey(0)

#SIFT detector initialization
#SIFT - Scale Invariant Feature Transform
sift = cv.SIFT_create()

#We need Descriptors and KeyPoint marks detection of both the images
#This can be done using detectAndCompute function in OpenCV
#For Image1
KeyPoint1, descriptor1 = sift.detectAndCompute(original_image,None)

#For Image2
KeyPoint2, descriptor2 = sift.detectAndCompute(cropped_image,None)

# BruteForce Matcher uses the distance calculation to match the features 
#between two images

#Initalize the BruteForce Matcher in OpenCV
bf = cv.BFMatcher()

#knnMatch finds the K nearest feature matching point and then allocates it to both image 
matches = bf.knnMatch(descriptor1,descriptor2,k=2)
print(matches)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv.drawMatchesKnn expects list of lists as matches.

FeatureMatched_image = cv.drawMatchesKnn(
    original_image,
    KeyPoint1,
    cropped_image,
    KeyPoint2,
    good,
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


plt.imshow(FeatureMatched_image),

plt.show()

cv.imshow('Watershed image',cropped_image)

cv.waitKey(0)

cv.destroyAllWindows()