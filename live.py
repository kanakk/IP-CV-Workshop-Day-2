# -*- coding: utf-8 -*-
"""
Author - Kanak Kawadiwale on Thu Nov  5 20:03:43 2020

@author: ag
"""

import cv2

cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("images/haarcascade_frontalface_default.xml")

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Converting the captures into grayscale images
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detecting faces in the web-cam stream
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		
	)

	print("Found {0} faces!".format(len(faces)))

	# Drawing a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)


	# Displaying the resulting frame
	cv2.imshow('Window', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

#free the camera resources once used
cap.release()

#Close all the running windows 
cv2.destroyAllWindows()