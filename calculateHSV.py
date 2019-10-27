import numpy as np
import cv2 as cv

#find HSV of tennis ball
red = np.uint8([[[141,182,0]]])

hsv_green = cv.cvtColor(red, cv.COLOR_BGR2HSV)

print(hsv_green)