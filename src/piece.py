class Piece:
    def __init__(self, number):
        self.number = number
        self.contour = []
        self.corners = []
    def setContour(self, contour):
        self.contour = contour
    def setCorners(self, corners):
        self.corners = corners
    