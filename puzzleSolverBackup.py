import numpy as np
import cv2
import time
from scipy import stats
import math


#num pieces in the puzzle
numPieces = 48
# increase for greater definition,, but will remove more color range
hueRange = 20
satRange = 100
valRange = 100
puzzle = 'src/puzzle2_1'

epsilon = 1000 #lower for smaller depth to be detected in defects

def getPieces( img ):
    #mask based on a range of colors for the background
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #reshape image into just 3 by num pixels list of the colors
    colors = img2.reshape(-1, 3)
    #find the most common color in that list
    bg = stats.mode(colors)[0][0]

    #make a mask based on the most common color
    mins = bg - np.array([hueRange, satRange, valRange])
    maxs = bg + np.array([hueRange, satRange, valRange])
    img3 = cv2.inRange(img2, mins, maxs)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img3 = cv2.dilate(img3, kernel, iterations = 2)
 
    #find contours in the mask
    contours, hierarchy = cv2.findContours(img3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #sort the contours by area, choose the biggest ones for the pieces
    # note that the largest contour is just the entire board
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[1:numPieces+1]

    return contours

def drawPieces( img, contours ):
    #put the contour areas on a blank background as white
    img2 = np.zeros_like(img)
    cv2.drawContours(img2, contours, -1, (255,255,255), thickness=-1)
    #take the white areas and include those areas from img
    img3 = img & img2
    return img3


def findLocalMinima( c1, c2 ):
    minimumDist = float('inf')
    minP1 = 0
    minP2 = 0
    for p1 in range(len(c1)):
        for p2 in range(len(c2)):
            x1 = c1[p1][0][0]
            y1 = c1[p1][0][1]
            x2 = c2[p2][0][0]
            y2 = c2[p2][0][1]
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < minimumDist:
                minimumDist = dist
                minP1 = p1
                minP2 = p2
    return minP1, minP2

def contourCenter( contour ):
    sumX = 0
    sumY = 0
    for p in contour:
        sumX += p[0][0]
        sumY += p[0][1]
    return sumX / len(contour), sumY / len(contour)

def drawClosures( img, contours ):
    for cnt in contours:
        hull = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(cnt,hull)
        xCen1, yCen1 = contourCenter(cnt)
        bigDefects = np.where(defects[:,:,3] > epsilon)[0]
        innerPoints = np.array([], dtype=np.int32)
        for i in range(len(bigDefects)):
            s,e,f,d = defects[bigDefects[i], 0]
            dist = math.sqrt((xCen1 - cnt[f][0][0])**2 + (yCen1 - cnt[f][0][1])**2)
            if dist < 55:
                innerPoints = np.append(innerPoints, bigDefects[i])

        outerPoints = np.setdiff1d(bigDefects, innerPoints)

        for i in range(len(outerPoints)):
            s,e,f,d = defects[outerPoints[i], 0]
            ps,pe,pf,pd = defects[outerPoints[i-1], 0]
            f1, f2 = findLocalMinima(cnt[pf - 20:pf + 20], cnt[f-20:f+20])
            f1 = pf - 20 + f1
            f2 = f - 20 + f2
            if f1 <= f2:
                cnt2 = cnt[f1:f2]
                cnt2 = np.concatenate((cnt2, cnt[f1][:,np.newaxis]))
            else:
                cnt2 = np.concatenate((cnt[f1:], cnt[:f2]))
                cnt2 = np.concatenate((cnt2, cnt[f1][:,np.newaxis]))
            
            area = cv2.contourArea(cnt2)
            perim = max(cv2.arcLength(cnt2, True), 1)
            circ = area / (perim**2)
            if abs(0.08 - circ) < 0.03 and area < 3000:
                cv2.drawContours(img, [cnt2], -1, (0,0,255), thickness=-1)
        
        for i in range(len(innerPoints)):
            s,e,f,d = defects[innerPoints[i], 0]
            cnt2 = cnt[s-30:f-30]
            if s-30 < 0:
                cnt2 = np.concatenate((cnt[s-30:], cnt[:f-30]))
            cnt3 = cnt[f+30:e+30]
            if e < f:
                cnt3 = np.concatenate((cnt[f+30:], cnt[:e+30]))
            s1, e1 = findLocalMinima(cnt2, cnt3)

            cv2.drawContours(img, [cnt[max(s-30+s1,0):min(f+30+e1,len(cnt) - 1)]], -1, (0,255,0), thickness=-1)

img = cv2.imread(f'{puzzle}.jpg')

contours = getPieces(img)
pieceImg = drawPieces(img, contours)
drawClosures(pieceImg, contours)
cv2.imwrite(f'{puzzle}Pieces.jpg', pieceImg)
