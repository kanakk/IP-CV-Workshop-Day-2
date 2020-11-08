# -*- coding: utf-8 -*-
"""
Author - Kanak Kawadiwale on Thu Nov  5 20:03:43 2020

@author: ag
"""

import pytesseract
from PIL import Image
import cv2 as cv

#Path towards the tessearact executable 
pytesseract.pytesseract.tesseract_cmd = r"D:\\OCR\\tesseract.exe"

print("\t\tTesseract file read success\n\n")

#read the image from which text is to be extracted 
img = cv.imread('images/text.jpg')

#convert the original image to grayscale image for better and faster processing
gray_converted = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

#apply the threshold to the image to remove the noise
ret, thresh1 = cv.threshold(gray_converted, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV) 

#Create a rectangular kernel inoreder to apply to the image. 
#Parameters - Shape of element, Size_of_element, anchor Position
rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (18, 18)) 

#Dilation of image would give you the clear and noise free image inorder to 
# detect the text without any noise. 
dilation = cv.dilate(thresh1, rect_kernel, iterations = 1)

#finally use the preprocessed image to detect the text in the image 

text_detected = pytesseract.image_to_string(gray_converted)

print("\t\tText Found.. Now Printing\n\n")
print(text_detected)