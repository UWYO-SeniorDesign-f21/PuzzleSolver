from scipy import stats
import cv2
import numpy as np
import math
import random

from piece import Piece
'''
The PieceCollection class stores all pieces. Includes functions to show the pieces
in order to make sure all the things are calculated correctly.
'''
class PieceCollection:
    def __init__(self, settings=[10, 50, 50, 12, 20, 32]):
        self.images = [] # images passed to the collection
        self.pieces = [] # pieces stored in the collection
        self.num_pieces_arr = [] # num pieces per image
        self.num_pieces_total = 0
        self.settings = settings

    '''
    Adds the pieces found in the image (filename) to the collection
    '''
    def addPieces(self, filename, num_pieces, color_spec="HSV"):
        image = cv2.imread(filename)
        h, w, _ = image.shape
        print(image.shape)

        # finds the contours and the labels for the pieces in the image
        contours = getContours(image, num_pieces, self.settings[:3], color_spec=color_spec)
        # image_glare_removed = removeGlare(contours, image)

        labels = getLabels(contours, len(self.images) + 1)
        # adds piece objects for each pair to the array of pieces
        for i, contour in enumerate(contours):
            label = labels[i]
            self.pieces.append(Piece(label, len(self.pieces), image, contour, self.settings[3:]))

        # adds the values to the arrays, total
        self.images.append(image)
        self.num_pieces_arr.append(num_pieces)
        self.num_pieces_total += num_pieces

    '''
    Saves images of each of the metrics for the pieces
    '''
    def getAllPiecesImage(self, with_details=True):
        w, h = max(self.num_pieces_arr), len(self.num_pieces_arr)

        image_dict = {}
        max_size = 0
        for piece in self.pieces:
            piece_image, _ = piece.getSubimage2(0, with_details=with_details, resize_factor=1)
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

    '''
    shows all the pieces
    '''
    def showPieceImages(self):
        for i, image in enumerate(self.images):
            print(i)
            pieces = [piece for piece in self.pieces if piece.image is image]
            contours = [piece.contour for piece in pieces]
            image_pieces = drawPieces(image, f'{i}', contours)
            image_pieces_2 = drawPieces(image, f'{i}', contours)
            image_zeros = np.zeros_like(image)
            image_zeros_2 = np.zeros_like(image)
            labels = [piece.label for piece in pieces]
            for piece in pieces:
                for k, corner in enumerate(piece.corners):
                    prev = piece.corners[k-1]
                    cv2.circle(image_pieces_2, (int(corner[1]), int(corner[2])), 20, (255, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.line(image_pieces_2, (int(corner[1]), int(corner[2])), (int(prev[1]), int(prev[2])),
                           (0,255,0), thickness=10)

                for k, edge in enumerate(piece.edges):
                    cv2.drawContours(image_pieces, edge.contour, -1, (255,255,255), thickness=10)
                    for j, d in enumerate(edge.distance_arr):
                        color = (128, max(min((d*2 + 128), 255), 0), 196)
                        cv2.drawContours(image_zeros, edge.contour, j, (int(color[0]), int(color[1]), int(color[2])), thickness=10)
                    for j, color in enumerate(edge.color_arr):
                        cv2.drawContours(image_zeros_2, edge.contour, 1 + j, (int(color[0]), int(color[1]), int(color[2])), thickness=15)

            h, w, _ = image_pieces.shape
            cv2.imwrite(f'image{i}pieces.png', image_pieces)
            cv2.imwrite(f'image{i}corners.png', image_pieces_2)
            cv2.imwrite(f'image{i}dists.png', image_zeros)
            cv2.imwrite(f'image{i}colors.png', cv2.cvtColor(image_zeros_2, cv2.COLOR_LAB2BGR))


def drawPieces( img, name, contours ):
    #put the contour areas on a blank background as white
    img2 = np.zeros_like(img)
    cv2.drawContours(img2, contours, -1, (255,255,255), thickness=-1)
    cv2.imwrite(f'{name}_pieces_mask.png', img2)
    #take the white areas and include those areas from img
    img3 = img & img2
    cv2.imwrite(f'{name}_isolated_pieces.png', img3)
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

'''
Calculates the contours for the pieces on the image
'''
def getContours(image, num_pieces, settings, color_spec="HSV"):
    color_range = settings
    # convert the image to the given color spectrum
    if color_spec == "HSV":
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_spec == "LAB":
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    else:
        image_hsv = image.copy()

    # reshape image into just 3 by num pixels list of the colors
    colors = image_hsv.reshape(-1, 3)
    colors = 5 * np.around(colors / 5)
    # find the most common color in that list
    background = stats.mode(colors)[0][0]

    #make a mask based on the most common color
    mins = background - np.array(color_range)
    maxs = background + np.array(color_range)
    mask = cv2.inRange(image_hsv, mins, maxs)

    # erode the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    mask = cv2.erode(mask, kernel, iterations=1)

    # fill the holes in the mask
    mask_contours, hierarchies = cv2.findContours(cv2.bitwise_not(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask_contours = [mask_contours[i] for i in range(len(mask_contours)) if hierarchies[0][i][3] < 0]
    mask_contours = sorted(mask_contours, key=cv2.contourArea, reverse=True)
    mask_contours = mask_contours[:num_pieces+1]

    mask_new = np.full_like(mask, 255)

    mask = cv2.drawContours(mask_new, mask_contours, -1, color=0, thickness=-1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    # dilate the mask
    mask = cv2.dilate(mask, kernel, iterations=5)

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

    # collection = PieceCollection(settings=[20, 40, 50, 16, 26, 64])
    # collection.addPieces('input/butterfly_01.jpg', 60)
    # collection.addPieces('input/butterfly_02.jpg', 77)
    # collection.addPieces('input/butterfly_03.jpg', 60)
    # collection.addPieces('input/butterfly_04.jpg', 66)
    # collection.addPieces('input/butterfly_05.jpg', 77)
    # collection.addPieces('input/butterfly_06.jpg', 60)
    # collection.addPieces('input/butterfly_07.jpg', 70)
    # collection.addPieces('input/butterfly_08.jpg', 43)

    # print(len([piece for piece in collection.pieces if piece.type == 'corner']))
    # print(len([piece for piece in collection.pieces if piece.type == 'side']))
    # print(len([piece for piece in collection.pieces if piece.type == 'middle']))

    # all_pieces = collection.getAllPiecesImage(with_details=True)
    # cv2.imwrite('butterfly_pieces.jpg', all_pieces)

    # collection = PieceCollection(settings=[10, 50, 50, 12, 30, 32])
    # collection.addPieces('input/tart_puzzle_01.jpg', 30)
    # collection.addPieces('input/tart_puzzle_02.jpg', 30)
    # collection.addPieces('input/tart_puzzle_03.jpg', 30)
    # collection.addPieces('input/tart_puzzle_04.jpg', 30)
    # collection.addPieces('input/tart_puzzle_05.jpg', 30)
    # collection.addPieces('input/tart_puzzle_06.jpg', 30)
    # collection.addPieces('input/tart_puzzle_07.jpg', 28)
    # collection.addPieces('input/tart_puzzle_08.jpg', 30)
    # collection.addPieces('input/tart_puzzle_09.jpg', 30)
    # collection.addPieces('input/tart_puzzle_10.jpg', 30)
    # collection.addPieces('input/tart_puzzle_11.jpg', 26)

    # collection = PieceCollection(settings=[10, 50, 50, 12, 20, 32])
    # collection.addPieces('input/travel_puzzle_01.jpg', 30)
    # collection.addPieces('input/travel_puzzle_02.jpg', 30)
    # collection.addPieces('input/travel_puzzle_03.jpg', 30)
    # collection.addPieces('input/travel_puzzle_04.jpg', 30)
    # collection.addPieces('input/travel_puzzle_05.jpg', 30)
    # collection.addPieces('input/travel_puzzle_06.jpg', 30)
    # collection.addPieces('input/travel_puzzle_07.jpg', 30)
    # collection.addPieces('input/travel_puzzle_08.jpg', 30)
    # collection.addPieces('input/travel_puzzle_09.jpg', 30)
    # collection.addPieces('input/travel_puzzle_10.jpg', 12)
    # collection.addPieces('input/travel_puzzle_11.jpg', 18)


    # collection = PieceCollection(settings=[20, 40, 50, 14, 16, 64])
    # collection.addPieces('input/feather_01.jpg', 40)
    # collection.addPieces('input/feather_02.jpg', 40)
    # collection.addPieces('input/feather_03.jpg', 40)
    # collection.addPieces('input/feather_04.jpg', 40)
    # collection.addPieces('input/feather_05.jpg', 40)
    # collection.addPieces('input/feather_06.jpg', 40)
    # collection.addPieces('input/feather_07.jpg', 40)
    # collection.addPieces('input/feather_08.jpg', 20)


    # collection = PieceCollection(settings=[20, 40, 40, 12, 20, 32])
    # collection.addPieces('input/animals_01.jpg', 77)
    # collection.addPieces('input/animals_02.jpg', 77)
    # collection.addPieces('input/animals_03.jpg', 77)
    # collection.addPieces('input/animals_04.jpg', 77)
    # collection.addPieces('input/animals_05.jpg', 77)
    # collection.addPieces('input/animals_06.jpg', 77)
    # collection.addPieces('input/animals_07.jpg', 47)
    # collection.addPieces('input/animals_08.jpg', 4)

    # collection = PieceCollection(settings=[10, 30, 30, 12, 16, 64])
    # collection.addPieces('input/owl2_01.jpg', 40, color_spec="HSV")
    # collection.addPieces('input/owl2_02.jpg', 40, color_spec="HSV")
    # collection.addPieces('input/owl2_03.jpg', 40, color_spec="HSV")
    # collection.addPieces('input/owl2_04.jpg', 40, color_spec="HSV")
    # collection.addPieces('input/owl2_05.jpg', 40, color_spec="HSV")
    # collection.addPieces('input/owl2_06.jpg', 40, color_spec="HSV")
    # collection.addPieces('input/owl2_07.jpg', 40, color_spec="HSV")
    # collection.addPieces('input/owl2_08.jpg', 20, color_spec="HSV")

    # collection = PieceCollection(settings=[10, 50, 50, 14, 20, 32])
    # collection.addPieces('input/pokemon_puzzle_1_01.png', 20)
    # collection.addPieces('input/pokemon_puzzle_1_02.png', 20)
    # collection.addPieces('input/pokemon_puzzle_1_03.png', 20)
    # collection.addPieces('input/pokemon_puzzle_1_04.png', 20)
    # collection.addPieces('input/pokemon_puzzle_1_05.png', 20)


    # collection = PieceCollection(settings=[10, 25, 30, 8, 14, 32])
    # collection.addPieces('input/300_01.png', 30)
    # collection.addPieces('input/300_02.png', 30)
    # collection.addPieces('input/300_03.png', 30)

    # collection = PieceCollection(settings=[30, 30, 30, 8, 18, 64])
    # collection.addPieces('input/market01.png', 48, color_spec="BGR")
    # collection.addPieces('input/market02.png', 48, color_spec="BGR")
    # collection.addPieces('input/market03.png', 48, color_spec="BGR")
    # collection.addPieces('input/market04.png', 48, color_spec="BGR")
    # collection.addPieces('input/market05.png', 37, color_spec="BGR")
    # collection.addPieces('input/market06.png', 36, color_spec="BGR")
    # collection.addPieces('input/market07.png', 48, color_spec="BGR")
    # collection.addPieces('input/market08.png', 48, color_spec="BGR")
    # collection.addPieces('input/market09.png', 48, color_spec="BGR")
    # collection.addPieces('input/market10.png', 43, color_spec="BGR")
    # collection.addPieces('input/market11.png', 48, color_spec="BGR")

    # collection = PieceCollection(settings=[15, 30, 50, 4, 30, 64])
    # collection.addPieces('input/pokemonBeach01.png', 48)
    # collection.addPieces('input/pokemonBeach02.png', 48)
    # collection.addPieces('input/pokemonBeach03.png', 48)
    # collection.addPieces('input/pokemonBeach04.png', 48)
    # collection.addPieces('input/pokemonBeach05.png', 48)
    # collection.addPieces('input/pokemonBeach06.png', 48)
    # collection.addPieces('input/pokemonBeach07.png', 42)
    # collection.addPieces('input/pokemonBeach08.png', 48)
    # collection.addPieces('input/pokemonBeach09.png', 48)
    # collection.addPieces('input/pokemonBeach10.png', 34)
    # collection.addPieces('input/pokemonBeach11.png', 34)
    # collection.addPieces('input/pokemonBeach12.png', 19)

    # collection = PieceCollection(settings=[30, 50, 50, 8, 18, 64])
    # collection.addPieces('input/waterfront01.png', 48)
    # collection.addPieces('input/waterfront02.png', 48)
    # collection.addPieces('input/waterfront03.png', 48)
    # collection.addPieces('input/waterfront04.png', 48)
    # collection.addPieces('input/waterfront05.png', 42)
    # collection.addPieces('input/waterfront06.png', 42)
    # collection.addPieces('input/waterfront07.png', 48)
    # collection.addPieces('input/waterfront08.png', 48)
    # collection.addPieces('input/waterfront09.png', 48)
    # collection.addPieces('input/waterfront10.png', 33)
    # collection.addPieces('input/waterfront11.png', 12)
    # collection.addPieces('input/waterfront12.png', 48)

    # collection = PieceCollection(settings=[30, 50, 50, 10, 20, 64])
    # collection.addPieces('input/donut01.png', 48)
    # collection.addPieces('input/donut02.png', 48)
    # collection.addPieces('input/donut03.png', 48)
    # collection.addPieces('input/donut04.png', 48)
    # collection.addPieces('input/donut05.png', 48)
    # collection.addPieces('input/donut06.png', 48)
    # collection.addPieces('input/donut07.png', 48)
    # collection.addPieces('input/donut08.png', 41)
    # collection.addPieces('input/donut09.png', 48)
    # collection.addPieces('input/donut10.png', 48)
    # collection.addPieces('input/donut11.png', 47)
    # collection.addPieces('input/donut12.png', 48)
    # collection.addPieces('input/donut13.png', 48)
    # collection.addPieces('input/donut14.png', 48)
    # collection.addPieces('input/donut15.png', 48)
    # collection.addPieces('input/donut16.png', 48)
    # collection.addPieces('input/donut17.png', 48)
    # collection.addPieces('input/donut18.png', 48)
    # collection.addPieces('input/donut19.png', 48)
    # collection.addPieces('input/donut20.png', 9)
    # collection.addPieces('input/donut21.png', 48)
    # collection.addPieces('input/donut22.png', 29)
    # collection.addPieces('input/donut23.png', 36)

    collection = PieceCollection(settings=[30, 50, 50, 10, 20, 64])
    collection.addPieces('input/patches01.png', 42)
    collection.addPieces('input/patches02.png', 48)
    collection.addPieces('input/patches03.png', 47)
    collection.addPieces('input/patches04.png', 48)
    collection.addPieces('input/patches05.png', 48)
    collection.addPieces('input/patches06.png', 48)
    collection.addPieces('input/patches07.png', 48)
    collection.addPieces('input/patches08.png', 47)
    collection.addPieces('input/patches09.png', 48)
    collection.addPieces('input/patches10.png', 48)
    collection.addPieces('input/patches11.png', 48)
    collection.addPieces('input/patches12.png', 48)
    collection.addPieces('input/patches13.png', 48)
    collection.addPieces('input/patches14.png', 48)
    collection.addPieces('input/patches15.png', 48)
    collection.addPieces('input/patches16.png', 14)
    collection.addPieces('input/patches17.png', 48)
    collection.addPieces('input/patches18.png', 48)
    collection.addPieces('input/patches19.png', 48)
    collection.addPieces('input/patches20.png', 48)
    collection.addPieces('input/patches21.png', 48)
    collection.addPieces('input/patches22.png', 47)
    collection.addPieces('input/patches23.png', 48)
    collection.addPieces('input/patches24.png', 48)
    collection.addPieces('input/patches25.png', 48)
    collection.addPieces('input/patches26.png', 48)
    collection.addPieces('input/patches27.png', 47)
    collection.addPieces('input/patches28.png', 48)
    collection.addPieces('input/patches29.png', 48)
    collection.addPieces('input/patches30.png', 48)
    collection.addPieces('input/patches31.png', 48)
    collection.addPieces('input/patches32.png', 48)
    collection.addPieces('input/patches33.png', 48)
    collection.addPieces('input/patches34.png', 47)
    collection.addPieces('input/patches35.png', 48)
    collection.addPieces('input/patches36.png', 48)
    collection.addPieces('input/patches37.png', 48)
    collection.addPieces('input/patches38.png', 48)
    collection.addPieces('input/patches39.png', 47)
    collection.addPieces('input/patches40.png', 48)
    collection.addPieces('input/patches41.png', 48)
    collection.addPieces('input/patches42.png', 48)
    collection.addPieces('input/patches43.png', 48)
    collection.addPieces('input/patches44.png', 48)
    collection.addPieces('input/patches45.png', 6)

    #collection.showPieceImages()
    #cv2.destroyAllWindows()

    print(len([piece for piece in collection.pieces if piece.type == 'corner']))
    print(len([piece for piece in collection.pieces if piece.type == 'side']))
    print(len([piece for piece in collection.pieces if piece.type == 'middle']))
    collection.showPieceImages()
    all_pieces = collection.getAllPiecesImage(with_details=True)
    cv2.imwrite('market_all_pieces.png', all_pieces)