import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while (1):

    _, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lowerRed = np.array([90, 100, 100])
    upperRed = np.array([100, 255, 255])

    mask = cv.inRange(hsv, lowerRed, upperRed)

    kernel = np.ones((5, 5), np.uint8)
    dilate = cv.dilate(mask, kernel, iterations=2)

    ret, thresh = cv.threshold(dilate, 15, 275, cv.THRESH_BINARY)

    M = cv.moments(thresh)

    if (M['m00'] > 0):

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 700, 700
    # put text and highlight the center
    cv.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
    # cv.putText(frame, "centroid", (cX - 60, cY - 60),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # parameters: input, contours to be passed in, draw all contours (-1) or index to a specific one, color, thickness
    img = cv.drawContours(frame, contours, -1, (0, 255, 0), 3)

    cv.imshow("Frame", frame)
    cv.imshow("hsv", hsv)
    cv.imshow("Mask", mask)

    k = cv.waitKey(5) & 0xff

    if k == 27:
        break

cv.destroyAllWindows()