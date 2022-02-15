from scipy import stats
import cv2
import numpy as np
import math
import random

from piece import Piece

class PieceCollection:
    def __init__(self):
        self.images = []
        self.pieces = []
        self.num_pieces_arr = []
        self.num_pieces_total = 0

    def addPieces(self, filename, num_pieces):
        image = cv2.imread(f'input/{filename}')
        h, w, _ = image.shape
        print(image.shape)

        # finds the contours and the labels for the pieces in the image
        contours = getContours(image, num_pieces)

        # resize pieces so that the average contour area is approximately equal to average of previous contour areas
        # avg_contour_size = np.mean([cv2.contourArea(contour) for contour in contours])
        # if len(self.pieces) > 0:
        #     avg_contour_size_prev = np.mean([cv2.contourArea(piece.contour) for piece in self.pieces])
        #     rescale_factor = math.sqrt(avg_contour_size_prev / avg_contour_size)
        #     print((int(rescale_factor*h), 
        #             int(rescale_factor*w)))
        #     image = cv2.resize(image, (int(rescale_factor*w), 
        #             int(rescale_factor*h)), interpolation = cv2.INTER_AREA)
        #     contours = getContours(image, num_pieces)
        #     print(avg_contour_size, avg_contour_size_prev)
        labels = getLabels(contours, len(self.images) + 1)
        # adds piece objects for each pair to the array of pieces
        for i, contour in enumerate(contours):
            label = labels[i]
            self.pieces.append(Piece(label, image, contour))
               
        # adds the values to the arrays, total
        self.images.append(image)
        self.num_pieces_arr.append(num_pieces)
        self.num_pieces_total += num_pieces
    
    def getAllPiecesImage(self, with_details=False):
        w, h = max(self.num_pieces_arr), len(self.num_pieces_arr)

        image_dict = {}
        max_size = 0
        for piece in self.pieces:
            piece_image = piece.getSubimage(0, with_details=with_details)
            image_dict[piece] = piece_image

            cv2.imshow('piece image', piece.getSubimage(0, with_details=True))
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


def drawPieces( img, contours ):
    #put the contour areas on a blank background as white
    img2 = np.zeros_like(img)
    cv2.drawContours(img2, contours, -1, (255,255,255),  lineType=cv2.FILLED, thickness=0)
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
        
def getContours(image, num_pieces):
    color_range = [10, 50, 50]
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

    #cv2.imshow('hsv',  cv2.rotate(cv2.resize(img3, (img.shape[1]//4, img.shape[0]//4), interpolation = cv2.INTER_AREA),
    #                    cv2.ROTATE_90_CLOCKWISE))
    #cv2.waitKey()

    # dilate the mask to get a slightly better fit
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=5)

    image2 = image_hsv.copy()

    h, w, _ = image.shape

    # mask2 = np.zeros((h+2, w+2), dtype=np.uint8)
    # mask2[1:-1, 1:-1] = cv2.bitwise_not(mask)
    # cv2.floodFill(image2, mask2, (10,10), 0, (20,30,50), (20,30,50))
    # cv2.imshow('image', cv2.resize(image2, (int(500 * (w / h)), 500), interpolation=cv2.INTER_AREA))
    # cv2.waitKey(1000)
    # mask = cv2.inRange(image2, (0,0,0), (0,0,0))
    # cv2.imshow('mask', cv2.resize(mask, (int(500 * (w / h)), 500), interpolation=cv2.INTER_AREA))
    # cv2.waitKey()
    # mask = cv2.dilate(mask, kernel, iterations=3)
    # mask = cv2.erode(mask, kernel, iterations=4)
    # mask = cv2.dilate(mask, kernel, iterations=1)

    #find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #cv2.imshow('hsv',  cv2.rotate(cv2.resize(img, (img.shape[1]//4, img.shape[0]//4), interpolation = cv2.INTER_AREA),
    #                    cv2.ROTATE_90_CLOCKWISE))
    #cv2.waitKey()

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
    collection = PieceCollection()
    # collection.addPieces('StarWarsPuzzle01.png', 6)
    # collection.addPieces('StarWarsPuzzle02.png', 6)
    # collection.addPieces('StarWarsPuzzle03.png', 6)
    # collection.addPieces('StarWarsPuzzle04.png', 6)
    # collection.addPieces('StarWarsPuzzle05.png', 6)
    # collection.addPieces('StarWarsPuzzle06.png', 6)
    # collection.addPieces('StarWarsPuzzle07.png', 6)
    # collection.addPieces('StarWarsPuzzle08.png', 6)

    # collection.addPieces('pokemon_puzzle_2_01.png', 20)
    # collection.addPieces('pokemon_puzzle_2_02.png', 20)
    # collection.addPieces('pokemon_puzzle_2_03.png', 20)
    # collection.addPieces('pokemon_puzzle_2_04.png', 20)
    # collection.addPieces('pokemon_puzzle_2_05.png', 20)

    # collection.addPieces('300_10.png', 30)
    # collection.addPieces('300_01.png', 30)
    # collection.addPieces('300_02.png', 30)
    # collection.addPieces('300_03.png', 30)
    # collection.addPieces('300_04.png', 30)
    # collection.addPieces('300_05.png', 30)
    # collection.addPieces('300_06.png', 30)
    # collection.addPieces('300_07.png', 30)
    # collection.addPieces('300_08.png', 30)
    # collection.addPieces('300_09.png', 30)
    collection.addPieces('tart_puzzle_01.jpg', 30)
    # collection.addPieces('tart_puzzle_02.jpg', 30)
    # collection.addPieces('tart_puzzle_03.jpg', 30)
    # collection.addPieces('tart_puzzle_04.jpg', 30)
    # collection.addPieces('tart_puzzle_05.jpg', 30)
    # collection.addPieces('tart_puzzle_06.jpg', 30)
    # collection.addPieces('tart_puzzle_07.jpg', 28)
    # collection.addPieces('tart_puzzle_08.jpg', 30)
    # collection.addPieces('tart_puzzle_09.jpg', 30)
    # collection.addPieces('tart_puzzle_10.jpg', 30)
    # collection.addPieces('tart_puzzle_11.jpg', 26)
    collection.addPieces('travel_puzzle_01.jpg', 30)
    collection.addPieces('travel_puzzle_02.jpg', 30)
    collection.addPieces('travel_puzzle_03.jpg', 30)
    collection.addPieces('travel_puzzle_04.jpg', 30)
    collection.addPieces('travel_puzzle_05.jpg', 30)
    collection.addPieces('travel_puzzle_06.jpg', 30)
    collection.addPieces('travel_puzzle_07.jpg', 30)
    collection.addPieces('travel_puzzle_08.jpg', 30)
    collection.addPieces('travel_puzzle_09.jpg', 30)
    collection.addPieces('travel_puzzle_10.jpg', 12)
    collection.addPieces('travel_puzzle_11.jpg', 18)

    # collection.addPieces('shining_01.jpg', 42)
    # collection.addPieces('shining_02.jpg', 42)
    # collection.addPieces('shining_03.jpg', 42)
    # collection.addPieces('shining_04.jpg', 42)
    # collection.addPieces('shining_05.jpg', 42)
    # collection.addPieces('shining_06.jpg', 42)
    # collection.addPieces('shining_07.jpg', 42)
    # collection.addPieces('shining_08.jpg', 42)
    # collection.addPieces('shining_09.jpg', 42)
    # collection.addPieces('shining_10.jpg', 42)

    collection.showPieceImages()
    all_pieces = collection.getAllPiecesImage(with_details=True)
    cv2.imwrite('starwars_all_pieces.png', all_pieces)

