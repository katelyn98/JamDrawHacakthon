import cv2 as cv
import numpy as np
from collections import deque
import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

cap = cv.VideoCapture(0)

#red, orange, yellow, green, blue, purple, pink
#i have RGB
#it goes BGR
#colors = [(38, 38, 255)]
colors = [(38, 38, 255), (19, 148, 249), (17, 250, 250), (17, 250, 48), (255, 207, 47), (255, 47, 165), (248, 107, 253)]
colorIndx = 0

redPts = [deque(maxlen=1000)]
orangePts = [deque(maxlen=1000)]
yellowPts = [deque(maxlen=1000)]
greenPts = [deque(maxlen=1000)]
bluePts = [deque(maxlen=1000)]
purplePts = [deque(maxlen=1000)]

redIndx = 0
orangeIndx = 0
yellowIndx = 0
greenIndx = 0
blueIndx = 0
purpleIndx = 0

def find_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for(percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        startX = endX

    return bar, color, percent


while(1):

    _, frame = cap.read()
    jamDraw = frame.copy()
    blackImg = np.zeros((512, 512, 3), np.uint8)
    flippedIMG = cv.flip(frame, 0)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lowerRed = np.array([90, 100, 100])
    upperRed = np.array([100, 255, 255])

    mask = cv.inRange(hsv, lowerRed, upperRed)

    kernel = np.ones((5, 5), np.uint8)
    dilate = cv.dilate(mask, kernel, iterations=2)

    ret, thresh = cv.threshold(dilate, 15, 275, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #parameters: input, contours to be passed in, draw all contours (-1) or index to a specific one, color, thickness
    img = cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
    center = None
    
    #drawing the colors to choose from
    points = [redPts, orangePts, yellowPts, greenPts, bluePts, purplePts]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                img2 = cv.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 23)

    #clear
    img = cv.rectangle(frame, (0, 60), (80, 90), (255, 255, 255), -1)
    #red
    img = cv.rectangle(frame, (80, 60), (160, 90), colors[0], -1)
    #orange
    img = cv.rectangle(frame, (160, 60), (240, 90), colors[1], -1)
    #yellow
    img = cv.rectangle(frame, (240, 60), (320, 90), colors[2], -1)
    #green
    img = cv.rectangle(frame, (320, 60), (400, 90), colors[3], -1)
    #blue
    img = cv.rectangle(frame, (400, 60), (480, 90), colors[4], -1)
    #purple
    img = cv.rectangle(frame, (480, 60), (560, 90), colors[5], -1)
    #take picture
    img = cv.rectangle(frame, (570, 390), (650, 420), (255, 255, 255), -1)

    if len(contours) > 0:

        M = cv.moments(thresh)
 
        if (M['m00'] > 0):

            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 700, 700

        cv.circle(frame, (cX, cY), 5, (255, 255, 255), -1)

        center = cX, cY

        if center[1] <= 90:
            if 0 <= center[0] <= 80:
                redPts = [deque(maxlen=1000)]
                orangePts = [deque(maxlen=1000)]
                yellowPts = [deque(maxlen=1000)]
                greenPts = [deque(maxlen=1000)]
                bluePts = [deque(maxlen=1000)]
                purplePts = [deque(maxlen=1000)]

                redIndx = 0
                orangeIndx = 0
                yellowIndx = 0
                greenIndx = 0
                blueIndx = 0
                purpleIndx = 0

            elif 80 <= center[0] <= 160:
                colorIndx = 0
            elif 160 <= center[0] <= 240:
                colorIndx = 1
            elif 240 <= center[0] <= 320:
                colorIndx = 2
            elif 320 <= center[0] <= 400:
                colorIndx = 3
            elif 400 <= center[0] <= 480:
                colorIndx = 4
            elif 480 <= center[0] <= 560:
                colorIndx = 5
        elif center[1] >= 390:
            if center[0] >= 570:

                cv.imwrite('jamDrawImg.jpg', img2)

                drawing = cv.imread('jamDrawImg.jpg')
                drawing = drawing[90:390, 0:700]
                #cv.imwrite('cropped.jpg', crop_img)
                #cv.imshow("cropped", crop_img)
                drawing = cv.cvtColor(drawing, cv.COLOR_BGR2RGB)

                drawing = drawing.reshape((drawing.shape[0] * drawing.shape[1], 3))
                clt = KMeans(n_clusters=3)
                clt.fit(drawing)

                hist = find_histogram(clt)
                bar, color, percent = plot_colors2(hist, clt.cluster_centers_)

                #print(hist)
                danceability = hist[0]
                valence = hist[1]
                tempo = hist[2]

                #plt.axis("off")
                #plt.imshow(bar)
                #plt.show()

        else:
            if colorIndx == 0:
                redPts[redIndx].appendleft(center)
            elif colorIndx == 1:
                orangePts[orangeIndx].appendleft(center)
            elif colorIndx == 2:
                yellowPts[yellowIndx].appendleft(center)
            elif colorIndx == 3:
                greenPts[greenIndx].appendleft(center)
            elif colorIndx == 4:
                bluePts[blueIndx].appendleft(center)
            elif colorIndx == 5:
                purplePts[purpleIndx].appendleft(center)
    else:
        redPts.append(deque(maxlen=1000))
        redIndx += 1
        orangePts.append(deque(maxlen=1000))
        orangeIndx += 1
        yellowPts.append(deque(maxlen=1000))
        yellowIndx += 1
        greenPts.append(deque(maxlen=1000))
        greenIndx += 1
        bluePts.append(deque(maxlen=1000))
        blueIndx += 1
        purplePts.append(deque(maxlen=1000))
        purpleIndx += 1


    cv.imshow("Frame", frame)
    #cv.imshow("hsv", hsv)
    #cv.imshow("Mask", mask)

    k = cv.waitKey(5) & 0xff

    if k == 27:
        break

cv.destroyAllWindows()