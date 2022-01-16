from scipy import stats
import cv2
import numpy as np
import math

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
        labels = getLabels(contours)
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


        
def getContours(image, num_pieces):
    color_range = [10, 10, 30]
    # convert the image to the hsv color spectrum
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # reshape image into just 3 by num pixels list of the colors
    colors = image_hsv.reshape(-1, 3)
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    #mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=5)
    mask = cv2.erode(mask, kernel, iterations=2)

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

def getLabels(contours):
    return np.zeros((len(contours),), dtype=int)

if __name__ == '__main__':
    collection = PieceCollection()
    collection.addPieces('puzzle3_02.jpg', 20)
    collection.addPieces('puzzle3_01.jpg', 20)
    collection.addPieces('puzzle3_03.jpg', 20)
    collection.addPieces('puzzle3_04.jpg', 20)
    collection.addPieces('puzzle3_05.jpg', 20)

    images = []
    for piece in collection.pieces:
        piece_image = piece.getSubimage(0, with_details=True)
        images.append(piece_image)

    sq_size = math.ceil(math.sqrt(len(images)))

    biggest_size = 0
    for image in images:
        image_size = max(image.shape[0], image.shape[1])
        if image_size > biggest_size:
            biggest_size = image_size
    
    all_pieces_image = np.zeros((biggest_size * sq_size, biggest_size * sq_size, 3), dtype=images[0].dtype)
    for i in range(sq_size):
        for j in range(sq_size):
            index = sq_size*i+j
            if index >= len(images):
                break
            h, w, _ = images[index].shape
            all_pieces_image[biggest_size*j:biggest_size*j + h, biggest_size*i:biggest_size*i + w] = images[index]

    cv2.imwrite(f'allPieces.jpg', all_pieces_image)