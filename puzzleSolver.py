import numpy as np
import cv2
import time

#num pieces in the puzzle
numPieces = 24

img = cv2.imread('puzzle1_1.jpg')
#mask based on a range of colors for the background
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite('tmp4.jpg', img2)
print(img2[4][4])
img3 = cv2.inRange(img2, np.array([0, 200, 200]), np.array([50, 255, 255]))
cv2.imwrite('tmp2.jpg', img3)

#find contours in the mask
contours, hierarchy = cv2.findContours(img3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#sort based on area, choose the numPieces largest ones
areas = np.array([cv2.contourArea(c) for c in contours])
pieces = np.flip(np.argsort(areas), axis=0)[1:1+numPieces]
pieceContours = [contours[piece] for piece in pieces]

#draw them on th OG image
cv2.drawContours(img, pieceContours, -1, (0, 0, 255), 3)
cv2.imwrite('tmp3.jpg', img)