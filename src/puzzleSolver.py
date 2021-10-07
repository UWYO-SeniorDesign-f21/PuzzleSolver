import puzzlePieces
import piece
import cv2
#from PieceObject import Piece

#num pieces in the puzzle 24 for puzzle 1 48 for puzzle 2
numPieces = 48
# increase for greater definition, but will remove more color range
# 3, 60, 60 for puzzle1 20,100,100 for puzzle 2
hueRange = 20
satRange = 100
valRange = 100

puzzleName = 'puzzle2_1'

pieces = puzzlePieces.PuzzlePieces(puzzleName, numPieces, hueRange, satRange, valRange)
pieces.findContours()
#img1 = pieces.showPieces()
pieces.findCorners()
img2 = pieces.showCorners()
cv2.imwrite(f'../{puzzleName}Corners.jpg', img2)
pieces.findEdges()
img3 = pieces.showEdges()
cv2.imwrite(f'../{puzzleName}Edges.jpg', img3)