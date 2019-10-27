import cv2 as cv
import numpy as np
from collections import deque

cap = cv.VideoCapture(0)

redIndx = 0
orangeIndx = 0
yellowIndx = 0
greenIndx = 0
blueIndx = 0
purpleIndx = 0
pinkIndx = 0

rPts = deque([0] * 512, maxlen=1000000000)
oPts = deque([0] * 512, maxlen=1000000000)
yPts = deque([0] * 512, maxlen=1000000000)
gPts = deque([0] * 512, maxlen=1000000000)
bPts = deque([0] * 512, maxlen=1000000000)
pPts = deque([0] * 512, maxlen=1000000000)
kPts = deque([0] * 512, maxlen=1000000000)

# red, orange, yellow, green, blue, purple, pink
# i have RGB
# it goes BGR
colors = [(38, 38, 255), (19, 148, 249), (17, 250, 250), (17, 250, 48), (255, 207, 47), (255, 47, 165), (248, 107, 253)]
colorIndx = 0

while (1):

    _, frame = cap.read()
    flippedIMG = cv.flip(frame, 0)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lowerRed = np.array([90, 100, 100])
    upperRed = np.array([100, 255, 255])

    mask = cv.inRange(hsv, lowerRed, upperRed)

    kernel = np.ones((5, 5), np.uint8)
    dilate = cv.dilate(mask, kernel, iterations=2)

    ret, thresh = cv.threshold(dilate, 15, 275, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # parameters: input, contours to be passed in, draw all contours (-1) or index to a specific one, color, thickness
    img = cv.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # drawing the colors to choose from

    # clear
    img = cv.rectangle(frame, (0, 60), (80, 90), (255, 255, 255), -1)
    # red
    img = cv.rectangle(frame, (80, 60), (160, 90), colors[0], -1)
    # orange
    img = cv.rectangle(frame, (160, 60), (240, 90), colors[1], -1)
    # yellow
    img = cv.rectangle(frame, (240, 60), (320, 90), colors[2], -1)
    # green
    img = cv.rectangle(frame, (320, 60), (400, 90), colors[3], -1)
    # blue
    img = cv.rectangle(frame, (400, 60), (480, 90), colors[4], -1)
    # purple
    img = cv.rectangle(frame, (480, 60), (560, 90), colors[5], -1)
    # pink
    img = cv.rectangle(frame, (560, 60), (640, 90), colors[6], -1)

    if len(contours) > 0:

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

        center = cX, cY

        if (cY <= 90):
            if 0 <= cX <= 80:
                redIndx = 0
                orangeIndx = 0
                yellowIndx = 0
                greenIndx = 0
                blueIndx = 0
                purpleIndx = 0
                pinkIndx = 0

                rPts = deque(maxlen=1000000000)
                oPts = deque(maxlen=1000000000)
                yPts = deque(maxlen=1000000000)
                gPts = deque(maxlen=1000000000)
                bPts = deque(maxlen=1000000000)
                pPts = deque(maxlen=1000000000)
                kPts = deque(maxlen=1000000000)

            elif 80 <= cX <= 160:
                colorIndx = 0  # Red
            elif 160 <= cX <= 240:
                colorIndx = 1  # Orange
            elif 240 <= cX <= 320:
                colorIndx = 2  # yellow
            elif 320 <= cX <= 400:
                colorIndx = 3  # Green
            elif 400 <= cX <= 480:
                colorIndx = 4  # Blue
            elif 480 <= cX <= 560:
                colorIndx = 5  # Purple
            elif 560 <= cX <= 640:
                colorIndx = 6  # pink
        else:
            if colorIndx == 0:
                rPts[redIndx].appendleft(center)
            elif colorIndx == 1:
                oPts[orangeIndx].appendleft(center)
            elif colorIndx == 2:
                yPts[yellowIndx].appendleft(center)
            elif colorIndx == 3:
                gPts[greenIndx].appendleft(center)
            elif colorIndx == 4:
                bPts[blueIndx].appendleft(center)
            elif colorIndx == 5:
                pPts[purpleIndx].appendleft(center)
            elif colorIndx == 6:
                kPts[pinkIndx].appendleft(center)
    else:
        rPts.append(deque(maxlen=1000000000))
        redIndx += 1

        oPts.append(deque(maxlen=1000000000))
        orangeIndx += 1

        yPts.append(deque(maxlen=1000000000))
        yellowIndx += 1

        gPts.append(deque(maxlen=1000000000))
        greenIndx += 1

        bPts.append(deque(maxlen=1000000000))
        blueIndx += 1

        pPts.append(deque(maxlen=1000000000))
        purpleIndx += 1

        kPts.append(deque(maxlen=1000000000))
        pinkIndx += 1

    points = [rPts, oPts, yPts, gPts, bPts, pPts, kPts]

    cv.imshow("Frame", frame)
    # cv.imshow("hsv", hsv)
    # cv.imshow("Mask", mask)

    k = cv.waitKey(5) & 0xff

    if k == 27:
        break

cv.destroyAllWindows()