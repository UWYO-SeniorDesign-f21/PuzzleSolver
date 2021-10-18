import numpy as np
import cv2
import time
import math
from scipy import stats
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
import random

import piece

class PuzzlePieces:
    def __init__(self, puzzleName, numPieces, hueRange, satRange, valRange):
        self.pieces = []
        self.puzzleName = puzzleName
        filename = f'../input/{puzzleName}.jpg'
        img = cv2.imread(filename)
        self.img = img
        self.numPieces = numPieces
        self.contours = []
        self.corners = []
        self.hueRange = hueRange
        self.satRange = satRange
        self.valRange = valRange
        self.solution = None
        self.solutionEdges = None
        self.rows = 6
        self.cols = 4
    def findContours(self):
        self.contours = getPieces(self.img, self.numPieces, self.hueRange, self.satRange, self.valRange)
        
        for i in range(len(self.contours)):
            piece2 = piece.Piece(i, self.img)
            piece2.setContour(self.contours[i])
            self.pieces.append(piece2)
    
    def showPieces(self):
        if len(self.contours) == 0:
            self.findContours()
        pieceImg = drawPieces(self.img, self.contours)
        cv2.imshow(f'showPieces {self.puzzleName}',  cv2.rotate(cv2.resize(pieceImg, (pieceImg.shape[1]//4, 
                    pieceImg.shape[0]//4), interpolation = cv2.INTER_AREA), cv2.ROTATE_90_CLOCKWISE))
        cv2.waitKey()
        return pieceImg
    
    def findCorners(self):
        if len(self.contours) == 0:
            self.findContours()
        self.corners = cornerDetection(self.contours)

        for i in range(len(self.corners)):
            piece = self.pieces[i]
            piece.setCorners(self.corners[i])
    
    def showCorners(self):
        if len(self.corners) == 0:
            self.findCorners()
        pieceImg = self.img.copy()
        for i in range(len(self.corners)):
            corner = self.corners[i]    
            for j in range(len(corner)):
                cv2.circle(pieceImg, (int(corner[j][1]), int(corner[j][2])), 10, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.line(pieceImg, (int(corner[j][1]), int(corner[j][2])), (int(corner[j-1][1]), int(corner[j-1][2])), 
                        (0,0,255), thickness=4)
        cv2.imshow(f'showCorners {self.puzzleName}',  cv2.rotate(cv2.resize(pieceImg, (pieceImg.shape[1]//4, 
                    pieceImg.shape[0]//4), interpolation = cv2.INTER_AREA), cv2.ROTATE_90_CLOCKWISE))
        cv2.waitKey()
        return pieceImg

    def findEdges(self):
        if len(self.corners) == 0:
            self.findCorners()
        for piece in self.pieces:
            piece.findEdges()

    def showEdges(self):
        if len(self.corners) == 0:
            self.findCorners()
        pieceImg = self.img.copy()
        for piece in self.pieces:
            piece.drawEdges(pieceImg)
        cv2.imshow(f'showEdges {self.puzzleName}',  cv2.rotate(cv2.resize(pieceImg, (pieceImg.shape[1]//4, 
                    pieceImg.shape[0]//4), interpolation = cv2.INTER_AREA), cv2.ROTATE_90_CLOCKWISE))
        cv2.waitKey()
        return pieceImg

    def findClosestPiece(self, piece1, edge1, pieces, puzzleArr, edgeArr, i, j, without):
            if i-1 > 0 and j != 0 and puzzleArr[i-1][j] != None:
                piece2 = puzzleArr[i-1][j]
                edge2 = edgeArr[i-1][j][2]
                edge, piece, dist = self.findClosestEdge([piece1, piece2], [edge1, edge2], pieces, without)
            else:
                edge, piece, dist = self.findClosestEdge([piece1], [edge1], pieces, without) 
            if piece == None:
                return None, None, float('inf')
            pieceResult = pieces[piece]
            return edge, pieceResult, dist

    def findClosestEdge(self, pieces1, edges, piecesAllowed, without):
        minDist = float('inf')
        minDistEdge = None
        minDistPiece = None
        for i in range(len(piecesAllowed)):
            if piecesAllowed[i] in pieces1:
                continue
            piece2 = piecesAllowed[i]
            edgeIndexes = np.array(range(4))
            withoutEdges = without[:,1][np.where(without[:,0] == piece2.number)]
            edgeIndexes = np.setdiff1d(edgeIndexes, withoutEdges)
            for edge2 in edgeIndexes:
                dist = 0
                for j in range(len(pieces1)):
                    piece1 = pieces1[j]
                    edge1 = edges[j]
                    dist1 = piece1.compareEdge(edge1, piece2, (edge2+j)%4)
                    dist = max(dist, dist1)
                if dist < minDist:
                    minDist = dist
                    minDistEdge = edge2
                    minDistPiece = i
        return minDistEdge, minDistPiece, minDist

    def puzzleSolver2(self):
        pieceImg = np.zeros_like(self.img)

        cornerPieces = []
        sidePieces = []
        middlePieces = []
        for piece in self.pieces:
            if piece.pieceLabel == 'corner':
                cornerPieces.append(piece)
            elif piece.pieceLabel == 'side':
                sidePieces.append(piece)
            else:
                middlePieces.append(piece)

        puzzleArr = np.empty((self.cols, self.rows), dtype=object) 
        edgeArr = np.empty((self.cols, self.rows, 4), dtype=np.int32)
        
        for i in range(self.cols):
            for j in range(self.rows):
                puzzleArr[i][j] = None
                edgeArr[i][j] = np.array([-1, -1, -1, -1])

        cLockLabels = cornerPieces[0].edgeLockLabels
        firstEdge = 0
        for i in range(len(cLockLabels)):
            if cLockLabels[i] == 'flat' and cLockLabels[i-1] == 'flat':
                firstEdge = (i-1)%4

        puzzleArr[0][0] = cornerPieces[0]
        # sorted edges, with up being the second edge on the corner, 
        # then going clockwise from there
        edgeArr[0][0] = np.array([(firstEdge+1)%4, (firstEdge+2)%4, (firstEdge+3)%4, firstEdge])

        without = np.array([[-1,-1]])
        
        for i in range(self.cols):
            for j in range(self.rows):
                piece1 = puzzleArr[i][j]
                edge1 = edgeArr[i][j][1]

                if piece1.pieceLabel == 'corner':
                    cornerPieces.remove(piece1)
                elif piece1.pieceLabel == 'side':
                    sidePieces.remove(piece1)
                else:
                    middlePieces.remove(piece1)

                if j == self.rows - 1:
                    break

                if j == self.rows - 2 and (i == 0 or i == self.cols - 1):
                    edge2, piece2, dist = self.findClosestPiece(piece1, edge1, cornerPieces, puzzleArr, edgeArr, i, j+1, without)
                elif j == self.rows - 2 or i == 0 or i == self.cols - 1:
                    edge2, piece2, dist = self.findClosestPiece(piece1, edge1, sidePieces, puzzleArr, edgeArr, i, j+1, without)
                else:
                    edge2, piece2, dist = self.findClosestPiece(piece1, edge1, middlePieces, puzzleArr, edgeArr, i, j+1, without) 

                if edge2 == None or piece2 == None:
                    cv2.waitKey()
                    self.solution = puzzleArr
                    self.solutionEdges = edgeArr
                    return pieceImg

                without = np.vstack((without, np.array([piece1.number, edge1])))
                without = np.vstack((without, np.array([piece2.number, edge2])))

                edgeArr[i,j+1] = np.array([(edge2+1)%4, 
                    (edge2+2)%4, (edge2+3)%4, edge2])
                puzzleArr[i][j+1] = piece2

                piece1.drawClosestEdge(pieceImg, edge1, piece2, edge2)
                cv2.imshow(f'closestEdge {self.puzzleName}',  cv2.rotate(cv2.resize(pieceImg, (pieceImg.shape[1]//4, 
                    pieceImg.shape[0]//4), interpolation = cv2.INTER_AREA), cv2.ROTATE_90_CLOCKWISE))
                k = cv2.waitKey(100)&0xff
                if k == 27:
                    self.solution = puzzleArr
                    self.solutionEdges = edgeArr
                    return pieceImg

            if i == self.cols - 1:
                break

            piece1 = puzzleArr[i][0]
            edge1 = edgeArr[i][0][2]

            if i == self.cols - 2:
                edge2, piece2, dist = self.findClosestPiece(piece1, edge1, cornerPieces, puzzleArr, edgeArr, i+1, 0, without)
            else:
                edge2, piece2, dist = self.findClosestPiece(piece1, edge1, sidePieces, puzzleArr, edgeArr, i+1, 0, without)

            if edge2 == None or piece2 == None:
                cv2.waitKey()
                self.solution = puzzleArr
                self.solutionEdges = edgeArr
                return pieceImg

            without = np.vstack((without, np.array([piece1.number, edge1])))
            without = np.vstack((without, np.array([piece2.number, edge2])))

            edgeArr[i+1,0] = np.array([edge2, 
                    (edge2+1)%4, (edge2+2)%4, (edge2+3)%4])
            puzzleArr[i+1,0] = piece2

            piece1.drawClosestEdge(pieceImg, edge1, piece2, edge2)
            cv2.imshow(f'closestEdge {self.puzzleName}',  cv2.rotate(cv2.resize(pieceImg, (pieceImg.shape[1]//4, 
                pieceImg.shape[0]//4), interpolation = cv2.INTER_AREA), cv2.ROTATE_90_CLOCKWISE))
            k = cv2.waitKey(100)&0xff
            if k == 27:
                self.solution = puzzleArr
                self.solutionEdges = edgeArr
                return pieceImg
        self.solution = puzzleArr
        self.solutionEdges = edgeArr
        return pieceImg

    def showPuzzleSolution(self):
        imgArray = [[[] for _ in range(len(self.solution[0]))] for _ in range(len(self.solution))]
        maxWidth = 0

        for i in range(len(self.solution)):
            for j in range(len(self.solution[i])):
                piece = self.solution[i][j]
                edgeUp = self.solutionEdges[i][j][0]
                if piece == None:
                    imgArray[i][j] = np.zeros((maxWidth, maxWidth, 3), dtype=self.img.dtype)
                    continue
                imgArray[i][j] = piece.cropAndRotate(edgeUp)
                if imgArray[i][j].shape[0] > maxWidth:
                    maxWidth = imgArray[i][j].shape[0]
        for i in range(len(self.solution)):
            for j in range(len(self.solution[i])):
                imgSize = imgArray[i][j].shape[0]
                t = (maxWidth - imgSize) // 2
                b = (maxWidth - imgSize) // 2
                imgSize1 = imgArray[i][j].shape[1]
                if t + b < maxWidth:
                    t = t + 1
                l = (maxWidth - imgSize1) // 2
                r = (maxWidth - imgSize1) // 2
                if l + r < maxWidth:
                    l = l + 1
                
                imgArray[i][j] = cv2.copyMakeBorder(imgArray[i][j], t, b, l, r, cv2.BORDER_CONSTANT, (0,0,0))
        
        pieceImg = None
        imgRow = None
        for i in range(len(self.solution)):
            for j in range(len(self.solution[i])):
                if j == 0:
                    imgRow = imgArray[i][0].copy()
                else:
                    imgRow = np.hstack((imgRow, imgArray[i][j]))
            if i == 0:
                pieceImg = imgRow.copy()
            else:
                pieceImg = np.vstack((pieceImg, imgRow))
        cv2.imshow('puzzleSolution', cv2.resize(pieceImg, (pieceImg.shape[1]//2, 
                    pieceImg.shape[0]//2), interpolation = cv2.INTER_AREA))
        cv2.waitKey(0)
        return pieceImg

def getPieces( img, numPieces, hueRange, satRange, valRange ):
    #mask based on a range of colors for the background
    #img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imshow('hsv',  cv2.rotate(cv2.resize(img2, (img.shape[1]//4, img.shape[0]//4), interpolation = cv2.INTER_AREA),
    #                    cv2.ROTATE_90_CLOCKWISE))
    cv2.waitKey()
    #reshape image into just 3 by num pixels list of the colors
    colors = img2.reshape(-1, 3)
    #find the most common color in that list
    bg = stats.mode(colors)[0][0]

    #make a mask based on the most common color
    mins = bg - np.array([hueRange, satRange, valRange])
    maxs = bg + np.array([hueRange, satRange, valRange])
    img3 = cv2.inRange(img2, mins, maxs)
    #cv2.imshow('hsv',  cv2.rotate(cv2.resize(img3, (img.shape[1]//4, img.shape[0]//4), interpolation = cv2.INTER_AREA),
    #                    cv2.ROTATE_90_CLOCKWISE))
    cv2.waitKey()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #img3 = cv2.erode(img3, kernel, iterations=2)
    img3 = cv2.dilate(img3, kernel, iterations=2)

    #find contours in the mask
    contours, hierarchy = cv2.findContours(img3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #cv2.imshow('hsv',  cv2.rotate(cv2.resize(img, (img.shape[1]//4, img.shape[0]//4), interpolation = cv2.INTER_AREA),
    #                    cv2.ROTATE_90_CLOCKWISE))
    cv2.waitKey()
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

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def closestDistPhi( phi1, phi2 ):
    if phi1 < 0:
        phi1 = 2*math.pi + phi1
    if phi2 < 0:
        phi2 = 2*math.pi + phi2
    dist = abs(phi1 - phi2)
    if dist > math.pi:
        if phi1 > math.pi:
            phi1 = -(2*math.pi - phi1)
        if phi2 > math.pi:
            phi2 = -(2*math.pi - phi2)
        dist = abs(phi1 - phi2)    
    return dist

def cornerDetection(contours):
    corners = np.empty((len(contours),4,3), dtype=np.int32)
    for i in range(len(contours)):
        cnt = contours[i]
        center = np.mean(cnt[:,0], axis=0)

        cntCen = np.empty_like(cnt)
        for j in range(len(cnt)):
            cntCen[j] = [cnt[j][0] - center]

        rho, phi = cart2pol(cntCen[:,0,0], cntCen[:,0,1])
        peaks, _ = find_peaks(rho, distance=50)
        
        if (peaks[0] + len(rho)) - peaks[-1] <= 50:
            if rho[peaks[0]] < rho[peaks[-1]]:
                peaks = peaks[1:]
            else:
                peaks = peaks[:-1]

        rhoMax = np.max(rho[peaks])

        diffLeft = rhoMax - rhoMax * rho[(peaks-10) % len(rho)] / rho[peaks]
        diffLeft[np.where(diffLeft < 0)] = 0
        diffRight = rhoMax - rhoMax * rho[(peaks+10) % len(rho)] / rho[peaks]
        diffRight[np.where(diffRight < 0)] = 0
        sharpness = diffLeft*diffRight
        
        #for smaller puzzle pieces
        #cornerPeaks = np.argsort(sharpness)[-4:]
        
        #for bigger puzzle pieces
        cornerPeaks = np.where(sharpness > 0)

        sharpness = sharpness[cornerPeaks]

        peaks = peaks[cornerPeaks]
                
        #plt.plot(rho, '.')
        #plt.plot(peaks, rho[peaks], 'x', c='orange')
        #plt.show()

        order = np.argsort([phi[peaks]])

        peaks = peaks[order][0]
        sharpness = sharpness[order][0]
        sharpnessSorted = np.flip(np.argsort(sharpness))

        peaks2 = np.empty((0,), dtype=np.int32)
        
        '''
        for smaller puzzles
        '''
        start = sharpnessSorted[0]
        peaks2 = np.append(peaks2, start)
        nextCorner = start
        while True:
            index = nextCorner
            phi1 = phi[peaks[index]]
            dists = np.empty((0,), dtype=np.float64)
            indexes = np.empty((0,), dtype=np.int32)
            for j in range(len(peaks)):
                if j in peaks2:
                    continue
                phi2 = phi[peaks[j]]
                dists = np.append(dists, closestDistPhi(phi1, phi2))
                indexes = np.append(indexes, j)
            if len(dists) > 0 and np.min(abs(dists - math.pi/2)) < math.pi/8:
                nextCorner = indexes[np.argmin(abs(dists - math.pi/2))]
                peaks2 = np.append(peaks2, nextCorner)
            elif len(dists) > 0:
                nextCorner = (nextCorner + 1) % len(peaks)
                peaks2 = np.append(peaks2, nextCorner)
            else:
                break
            if len(peaks2) >= 4:
                break
        peaks = peaks[peaks2]
        '''
        end section
        '''
        order = np.argsort([phi[peaks]])
        peaks = peaks[order][0]
        
        cornerX = cnt[:,0,0][peaks]
        cornerY = cnt[:,0,1][peaks]
        
        corners[i,:,0] = peaks
        corners[i,:,1] = cornerX
        corners[i,:,2] = cornerY
    return corners