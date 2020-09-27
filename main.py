import cv2 as cv
import numpy as np

# read image file
img = cv.imread("./data/image.jpg")

# show it
cv.imshow('image',img)
cv.waitKey(0)
