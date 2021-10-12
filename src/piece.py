import numpy as np
import cv2

class Piece:
    def __init__(self, number):
        self.number = number
        self.contour = []
        self.corners = []
        self.edges = []
        self.edgeContours = []
        self.edgeLockLabels = []
        self.edgeLockLocs = []
    def setContour(self, contour):
        self.contour = contour
    def setCorners(self, corners):
        self.corners = corners
    def findEdges(self):
        epsilon = 30
        if len(self.corners) == 0 or len(self.contour) == 0:
            return
        for i in range(len(self.corners)):
            index1 = self.corners[i-1][0]
            pnt1 = self.corners[i-1][1:]
            index2 = (self.corners[i][0] + 1) % len(self.contour)
            pnt2 = self.corners[i][1:]
            if index1 < index2:
                edgeContour = self.contour[index1:index2]
            else:
                edgeContour = np.concatenate((self.contour[index1:], self.contour[:index2]))

            self.edgeContours.append(edgeContour)

            # distance from each contour point to the line between corners
            d = -np.cross(pnt2-pnt1,edgeContour-pnt1)/np.linalg.norm(pnt2-pnt1)

            furthestPoint = np.max(abs(d))
            if furthestPoint < epsilon:
                self.edgeLockLabels.append('flat')
                self.edgeLockLocs.append(np.argmin(d))
            elif np.max(d) == furthestPoint:
                self.edgeLockLabels.append('out')
                self.edgeLockLocs.append(np.argmax(d))
            elif np.min(d) == -furthestPoint:
                self.edgeLockLabels.append('in')
                self.edgeLockLocs.append(np.argmin(d))   

            self.edges.append(d)
    
    def drawEdges(self, img):
        if len(self.edgeContours) == 0:
            self.find(edges)
        for i in range(len(self.edgeContours)):
            if self.edgeLockLabels[i] == 'in':
                color = (255,0,0)
                color2 = (255,0,200)
            elif self.edgeLockLabels[i] == 'flat':
                color = (255,0,255)
                color2 = (255,100,255)
            elif self.edgeLockLabels[i] == 'out':
                color = (0,0,255)
                color2 = (0,200,255)
            cv2.drawContours(img, self.edgeContours[i], -1, color, thickness=8)
            location = self.edgeContours[i][self.edgeLockLocs[i]][0]
            if self.edgeLockLabels[i] != 'flat':
                cv2.circle(img, (int(location[0]), int(location[1])), 10, color2, thickness=-1, lineType=cv2.FILLED)