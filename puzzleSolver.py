import numpy as np
import cv2
import time
from scipy import stats

#num pieces in the puzzle
numPieces = 24

def getPieces( img ):
    #mask based on a range of colors for the background
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #reshape image into just 3 by num pixels list of the colors
    colors = img2.reshape(-1, 3)
    #find the most common color in that list
    bg = stats.mode(colors)[0][0]

    #make a mask based on the most common color
    mins = bg - np.array([5, 100, 255])
    maxs = bg + np.array([5, 100, 255])
    img3 = cv2.inRange(img2, mins, maxs)

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

img = cv2.imread('puzzle1_1.jpg')

contours = getPieces(img)
pieceImg = drawPieces(img, contours)
cv2.imwrite('puzzle1_1Pieces.jpg', pieceImg)
