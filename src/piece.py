import numpy as np
import cv2

class Piece:
    def __init__(self, number, img):
        self.number = number
        self.img = img
        self.contour = []
        self.corners = []
        self.edges = []
        self.edgeContours = []
        self.edgeLockLabels = []
        self.edgeLockLocs = []
        self.edgeColors = []
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
            self.edgeColors.append(self.findEdgeColors(i))
    
    def drawEdges(self, img, edges=range(4)):
        if len(self.edgeContours) == 0:
            self.find(edges)
        for i in edges:
            if self.edgeLockLabels[i] == 'in':
                color = (255,0,0)
                color2 = (255,0,200)
            elif self.edgeLockLabels[i] == 'flat':
                color = (255,0,255)
                color2 = (255,100,255)
            elif self.edgeLockLabels[i] == 'out':
                color = (0,0,255)
                color2 = (0,200,255)
            cv2.drawContours(img, self.edgeContours[i], -1, color, thickness=3)
            location = self.edgeContours[i][self.edgeLockLocs[i]][0]
            if self.edgeLockLabels[i] != 'flat':
                cv2.circle(img, (int(location[0]), int(location[1])), 10, color2, thickness=-1, lineType=cv2.FILLED)
    
    def drawColorEdges(self, img, edges=range(4)):
        if len(self.edgeContours) == 0:
            self.find(edges)
        for i in edges:
            for j in range(1, len(self.edgeContours[i])-2):
                if j % 10 == 0:
                    cv2.circle(img, (self.edgeContours[i][j][0][0], self.edgeContours[i][j][0][1]), 10,
                        self.edgeColors[i][j], thickness=-1, lineType=cv2.FILLED)

    def findEdgeColors(self, edge):
        start, end = 5,10
        ec = self.edgeContours[edge]
        # want line perpindicular to contour at every point
        # then want average on that line for some pixels of color
        pPrev = ec[0:-2,0]
        pCurr = ec[1:-1,0]
        pNext = ec[2:,0]
        ecPerp = np.empty_like(pPrev)
        for i in range(len(pPrev)):
            dx = -pNext[i,0] + pPrev[i,0]
            dy = -pNext[i,1] + pPrev[i,1]
            ecPerp[i] = [dy, -dx]
        ecPerp = ecPerp / np.linalg.norm(ecPerp, axis=1)[:,np.newaxis]
        colorEc = np.zeros((len(ecPerp), 3))
        for i in range(len(ecPerp)):
            colorSum = np.zeros((3,), dtype=int)
            for j in range(start, end):
                pt = (pCurr[i] + j*ecPerp[i]).astype(int)
                colorSum += self.img[pt[1], pt[0], :]
            colorEc[i] = colorSum // (end - start)
        return colorEc

    def findClosestEdge(self, edge, pieces, without):
        minDist = float('inf')
        minDistEdge = None
        minDistPiece = None
        for i in range(len(pieces)):
            if pieces[i] == self:
                continue
            piece2 = pieces[i]
            edgeIndexes = np.array(range(4))
            withoutEdges = without[:,1][np.where(without[:,0] == i)]
            edgeIndexes = np.setdiff1d(edgeIndexes, withoutEdges)
            for edge2 in edgeIndexes:
                dist = self.compareEdge(edge, piece2, edge2)
                if dist < minDist:
                    minDist = dist
                    minDistEdge = edge2
                    minDistPiece = i
        return minDistEdge, minDistPiece, minDist

    def drawClosestEdge(self, img, edge1, piece2, edge2):
        self.drawColorEdges(img, [edge1])
        piece2.drawColorEdges(img, [edge2])
        loc1 = self.edgeContours[edge1][self.edgeLockLocs[edge1]][0]
        loc2 = piece2.edgeContours[edge2][piece2.edgeLockLocs[edge2]][0]
        cv2.line(img, (int(loc1[0]), int(loc1[1])), (int(loc2[0]), int(loc2[1])), (0,255,0), 6)
    
    def compareEdge(self, edge1, piece2, edge2):
        if self.edgeLockLabels[edge1] == 'flat' or piece2.edgeLockLabels[edge2] == 'flat':
            return float('inf')
        
        badEdgeAddOn = 0
        
        # putting edge pieces w/ edge pieces
        if self.edgeLockLabels[(edge1 + 1) % 4] == 'flat' and piece2.edgeLockLabels[(edge2 - 1)%4] != 'flat':
            badEdgeAddOn = 10000
        if self.edgeLockLabels[edge1 - 1] == 'flat' and piece2.edgeLockLabels[(edge2 + 1) % 4] != 'flat':
            badEdgeAddOn = 10000
        if self.edgeLockLabels[(edge1 + 1) % 4] != 'flat' and piece2.edgeLockLabels[edge2 - 1] == 'flat':
            badEdgeAddOn = 10000
        if self.edgeLockLabels[edge1 - 1] != 'flat' and piece2.edgeLockLabels[(edge2 + 1) % 4] == 'flat':
            badEdgeAddOn = 10000

        if len(self.edges[edge1]) <= len(piece2.edges[edge2]):
            ec1 = -self.edges[edge1]
            eColor1 = self.edgeColors[edge1]
            ec2 = np.flip(piece2.edges[edge2])
            eColor2 = np.flip(piece2.edgeColors[edge2], axis=0)
        else:
            ec2 = -self.edges[edge1]
            eColor2 = self.edgeColors[edge1]
            ec1 = np.flip(piece2.edges[edge2])
            eColor1 = np.flip(piece2.edgeColors[edge2], axis=0)

        minDist = float('inf')
        minColorDist = float('inf')
        minI = 0

        if len(ec2) == len(ec1):
            minDist = np.sum(abs(ec2 - ec1))
        else:  
            for i in range(len(ec2) - len(ec1)):
                dist = np.sum(abs(ec2[i:len(ec1) + i] - ec1))
                dist += (len(ec2) - len(ec1))**2
                if dist < minDist:
                    minDist = dist
                    minI = i
        if minDist < float('inf'):
            minColorDist = np.sum(np.linalg.norm(eColor2[max(minI,1)-1:len(eColor1)+max(minI,1)-1] - eColor1))

        return minDist + minColorDist/2 + badEdgeAddOn
            