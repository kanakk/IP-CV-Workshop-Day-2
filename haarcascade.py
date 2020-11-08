# -*- coding: utf-8 -*-
"""
Author - Kanak Kawadiwale on Thu Nov  5 20:03:43 2020

@author: ag
"""

import cv2

#Import the cascade file for Frontface detection
face_cascade = cv2.CascadeClassifier('images/haarcascade_frontalface_default.xml')

#Import the cascade file for Eye detection
eye_cascade = cv2.CascadeClassifier('images/haarcascade_eye.xml')


img = cv2.imread('images/smile.jpg')

#Preprocessing and converting the orgiginal image to grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#Parameters - Image Source, ScaleFactor, minNeighbours
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:

    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,90,178),2)
    
    roi_gray = gray[y:y+h, x:x+w]

    roi_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray)

    for (ex,ey,ew,eh) in eyes:

        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()