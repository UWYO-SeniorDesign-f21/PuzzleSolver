from scipy import stats
import cv2
import numpy as np
import math
import random

from piece import Piece

class PieceCollection:
    def __init__(self, settings=[10, 50, 50, 12, 20, 32]):
        self.images = []
        self.pieces = []
        self.num_pieces_arr = []
        self.num_pieces_total = 0
        self.settings = settings

    def addPieces(self, filename, num_pieces):
        image = cv2.imread(filename)
        h, w, _ = image.shape
        print(image.shape)

        # finds the contours and the labels for the pieces in the image
        contours = getContours(image, num_pieces, self.settings[:3])

        labels = getLabels(contours, len(self.images) + 1)
        # adds piece objects for each pair to the array of pieces
        for i, contour in enumerate(contours):
            label = labels[i]
            self.pieces.append(Piece(label, image, contour, self.settings[3:]))
               
        # adds the values to the arrays, total
        self.images.append(image)
        self.num_pieces_arr.append(num_pieces)
        self.num_pieces_total += num_pieces
    
    def getAllPiecesImage(self, with_details=False):
        w, h = max(self.num_pieces_arr), len(self.num_pieces_arr)

        image_dict = {}
        max_size = 0
        for piece in self.pieces:
            piece_image = piece.getSubimage(0, with_details=with_details, resize_factor=2)
            image_dict[piece] = piece_image

            cv2.imshow('piece image', piece_image)
            cv2.waitKey()

            piece_image_size = max(piece_image.shape)
            if piece_image_size > max_size:
                max_size = piece_image_size
        
        pieces_image = np.zeros((max_size * h, max_size * w, 3), dtype=self.images[0].dtype)
        h, w, _ = pieces_image.shape
        index = 0
        for i, num_pieces in enumerate(self.num_pieces_arr):
            for j in range(num_pieces):
                piece = self.pieces[index + j]
                piece_image = image_dict[piece]

                ph, pw, _ = piece_image.shape

                x_coord = j*max_size
                y_coord = i*max_size

                pieces_image[y_coord:y_coord + ph, x_coord:x_coord + pw] = piece_image
            index += num_pieces

        return pieces_image

    def showPieceImages(self):
        for i, image in enumerate(self.images):
            pieces = [piece for piece in self.pieces if piece.image is image]
            contours = [piece.contour for piece in pieces]
            image_pieces = drawPieces(image, contours)
            labels = [piece.label for piece in pieces]
            image_pieces = showLabels(image_pieces, contours, labels)
            h, w, _ = image_pieces.shape
            cv2.imshow(f'image {i}', cv2.resize(image_pieces, (int(500 * (w / h)), 500), interpolation=cv2.INTER_AREA))
            cv2.waitKey()
            cv2.imwrite(f'image{i}shining.png', image_pieces)


def drawPieces( img, contours ):
    #put the contour areas on a blank background as white
    img2 = np.zeros_like(img)
    cv2.drawContours(img2, contours, -1, (255,255,255), thickness=-1)
    #take the white areas and include those areas from img
    img3 = img & img2
    return img3

def showLabels( img, contours, labels ):
    font_size = 2
    font_thickness = 5
    for i, contour in enumerate(contours):
        center = np.mean(contour[:,0], axis=0).astype(int)
        textsize = cv2.getTextSize(str(labels[i]), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)[0]

        # get coords based on boundary
        textX = center[0] - int(textsize[0] / 2)
        textY = center[1] + int(textsize[1] / 2)

        rect_pos1 = (textX - 20, center[1] - int(textsize[1] / 2) - 20)
        rect_pos2 = (center[0] + int(textsize[0] / 2) + 20, textY + 20)
        cv2.rectangle(img, rect_pos1, rect_pos2, (255,255,255), -1)
        cv2.putText(img, str(labels[i]), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,0), font_thickness, 3)
    return img
        
def getContours(image, num_pieces, settings):
    color_range = settings
    # convert the image to the hsv color spectrum
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # reshape image into just 3 by num pixels list of the colors
    colors = image_hsv.reshape(-1, 3)
    colors = 5 * np.around(colors / 5)
    # find the most common color in that list
    background = stats.mode(colors)[0][0]

    #make a mask based on the most common color
    mins = background - np.array(color_range)
    maxs = background + np.array(color_range)
    mask = cv2.inRange(image_hsv, mins, maxs)

    # dilate the mask to get a slightly better fit
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=5)
    
    #cv2.imshow("mask", cv2.resize(mask, (500, 500), interpolation=cv2.INTER_AREA))
    #cv2.waitKey(0)

    #find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # sort the contours by area, choose the biggest ones for the pieces
    # note that the largest contour is just the entire board
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[1:num_pieces+1]
    
    return contours

def getLabels(contours, image_num):
    labels = []
    for i, contour in enumerate(contours):
        labels.append(f'{image_num}: {i}')
    return labels
if __name__ == '__main__':

    collection = PieceCollection(settings=[10, 50, 50, 12, 20, 32])
    collection.addPieces('input/butterfly_01.jpg', 60)
    #collection.showPieceImages()
    #cv2.destroyAllWindows()

    print(len([piece for piece in collection.pieces if piece.type == 'corner']))
    print(len([piece for piece in collection.pieces if piece.type == 'side']))
    print(len([piece for piece in collection.pieces if piece.type == 'middle']))

    all_pieces = collection.getAllPiecesImage(with_details=True)
    cv2.imwrite('shining_all_pieces.png', all_pieces)