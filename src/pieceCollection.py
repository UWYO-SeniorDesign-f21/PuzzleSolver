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
            self.pieces.append(Piece(label, image, contour, self.settings[3:]))

        # adds the values to the arrays, total
        self.images.append(image)
        self.num_pieces_arr.append(num_pieces)
        self.num_pieces_total += num_pieces

    def getAllPiecesImage(self, with_details=True):
        w, h = max(self.num_pieces_arr), len(self.num_pieces_arr)

        image_dict = {}
        max_size = 0
        for piece in self.pieces:
            piece_image = piece.getSubimage(0, with_details=with_details, resize_factor=1)
            image_dict[piece] = piece_image

            # cv2.imshow('piece image', piece_image)
            # cv2.waitKey()

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
            # cv2.imshow(f'image {i}', cv2.resize(image_pieces, (int(500 * (w / h)), 500), interpolation=cv2.INTER_AREA))
            # cv2.waitKey()
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

def getContours(image, num_pieces, settings, color_spec="HSV"):
    color_range = settings
    # convert the image to the hsv color spectrum
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

    # cv2.imshow('in range', cv2.resize(mask, (500, int(500 * (mh/mw))), interpolation=cv2.INTER_AREA))
    # cv2.imwrite('inrange.jpg', mask)

    # dilate the mask to get a slightly better fit
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    mask = cv2.erode(mask, kernel, iterations=1)

    # cv2.imshow('erode', cv2.resize(mask, (500, int(500 * (mh/mw))), interpolation=cv2.INTER_AREA))
    # cv2.imwrite('erode.jpg', mask)

    mask_contours, hierarchies = cv2.findContours(cv2.bitwise_not(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask_contours = [mask_contours[i] for i in range(len(mask_contours)) if hierarchies[0][i][3] < 0]
    mask_contours = sorted(mask_contours, key=cv2.contourArea, reverse=True)
    mask_contours = mask_contours[:num_pieces+1]

    # mask_just_edges = np.full_like(mask, 255)
    # mask_just_edges = cv2.drawContours(mask_just_edges, mask_contours, -1, color=0, thickness=4)

    # cv2.imshow('contours', cv2.resize(mask_just_edges, (500, int(500 * (mh/mw))), interpolation=cv2.INTER_AREA))

    mask_new = np.full_like(mask, 255)

    mask = cv2.drawContours(mask_new, mask_contours, -1, color=0, thickness=-1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    mask = cv2.dilate(mask, kernel, iterations=5)

    # cv2.imshow('final', cv2.resize(mask, (500, int(500 * (mh/mw))), interpolation=cv2.INTER_AREA))
    # cv2.waitKey(0)
    # cv2.imwrite('final_mask.jpg', mask)

    # cv2.imshow("mask", cv2.resize(mask, (500, 500), interpolation=cv2.INTER_AREA))
    # cv2.waitKey(0)

    #find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # sort the contours by area, choose the biggest ones for the pieces
    # note that the largest contour is just the entire board
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[1:num_pieces+1]

    return contours

def removeGlare(contours, image):
    contour_mask1 = np.zeros(image.shape[:2], dtype=np.uint8)
    contour_mask1 = cv2.drawContours(contour_mask1, contours, -1, color=255, thickness=40)
    contour_mask2 = np.zeros(image.shape[:2], dtype=np.uint8)
    contour_mask2 = cv2.drawContours(contour_mask2, contours, -1, color=255, thickness=-1)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    # contour_mask2 = cv2.erode(contour_mask2, kernel, iterations=1)

    contour_mask = cv2.bitwise_and(contour_mask1, contour_mask2)
    image_masked = np.zeros_like(image)
    image_masked[contour_mask > 0] = image[contour_mask > 0]
    

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    # contour_mask = cv2.erode(contour_mask, kernel, iterations=1)

    # cv2.imshow('im', cv2.resize(image_masked, (425, 550)))

    image_masked[contour_mask > 0] = image[contour_mask > 0]
    image_hsv = cv2.cvtColor(image_masked, cv2.COLOR_BGR2LAB).astype(np.uint8)

    values = image_hsv[:,:,0]

    bright_spots = cv2.inRange(values, 235, 255)

    # cv2.imshow('bs', cv2.resize(bright_spots, (425, 550)))


    bright_spots_dilated = cv2.dilate(bright_spots, kernel, iterations=1)

    bright_contours, _ = cv2.findContours(bright_spots_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    glare_contours = []
    for contour in bright_contours:
        good_contour=True
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            print('????')
        if contour_mask1[cy, cx] == 0:
            good_contour=False
        else: 
            for pt in contour:
                if contour_mask1[pt[0][1], pt[0][0]] == 0:
                    good_contour=False
                    break
        if good_contour:
            glare_contours.append(contour)

    image_glare_removed = image.copy()
    contour_mask_check = cv2.erode(contour_mask2, kernel, iterations=1)

    for i, contour in enumerate(glare_contours):
        this_contour_mask = np.zeros_like(contour_mask)
        this_contour_mask = cv2.drawContours(this_contour_mask, glare_contours, i, color=255, thickness=-1)
        this_contour_mask_inflated = cv2.dilate(this_contour_mask, kernel, 3)
        this_contour_mask_double_inflated = cv2.dilate(this_contour_mask_inflated, kernel, 3)

        check_region = cv2.bitwise_xor(this_contour_mask_double_inflated, this_contour_mask_inflated)
        check_region = cv2.bitwise_and(check_region, contour_mask_check)
        
        color_avg = cv2.mean(image, mask=check_region)
        color = np.array([color_avg[0], color_avg[1], color_avg[2]])

        place_region = cv2.bitwise_and(this_contour_mask_double_inflated, contour_mask_check)
        filled = np.bitwise_not(place_region)

        kernel2 = np.ones((2,2)).astype(int)

        x, y, w, h = cv2.boundingRect(contour)

        while np.any(filled == 0):
            check_region = cv2.dilate(check_region, kernel2, iterations=1)
            expand_points = np.argwhere(cv2.bitwise_and(check_region, place_region) > 0)
            for point in expand_points:
                filled = cv2.circle(filled, (point[1], point[0]), 3, 255, -1)
                place_region = cv2.circle(place_region, (point[1], point[0]), 3, 0, -1)
                color = tuple ([int(val) for val in image_glare_removed[point[0], point[1]]])
                image_glare_removed = cv2.circle(image_glare_removed, (point[1], point[0]), 3, color,  -1)

        # image_glare_removed[place_region > 0] = color

    cv2.imwrite('image_glare_removed.jpg', image_glare_removed)

    # cv2.imshow('gs', cv2.resize(glare_spots, (425, 550)))
    # cv2.waitKey(0)            

    return image_glare_removed



def getLabels(contours, image_num):
    labels = []
    for i, contour in enumerate(contours):
        labels.append(f'{image_num}: {i}')
    return labels
if __name__ == '__main__':

    # collection = PieceCollection(settings=[20, 40, 50, 12, 20, 32])
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

    collection = PieceCollection(settings=[30, 50, 50, 10, 20, 64])
    collection.addPieces('input/donut01.png', 48)
    collection.addPieces('input/donut02.png', 48)
    collection.addPieces('input/donut03.png', 48)
    collection.addPieces('input/donut04.png', 48)
    collection.addPieces('input/donut05.png', 48)
    collection.addPieces('input/donut06.png', 48)
    collection.addPieces('input/donut07.png', 48)
    collection.addPieces('input/donut08.png', 41)
    collection.addPieces('input/donut09.png', 48)
    collection.addPieces('input/donut10.png', 48)
    collection.addPieces('input/donut11.png', 47)
    collection.addPieces('input/donut12.png', 48)
    collection.addPieces('input/donut13.png', 48)
    collection.addPieces('input/donut14.png', 48)
    collection.addPieces('input/donut15.png', 48)
    collection.addPieces('input/donut16.png', 48)
    collection.addPieces('input/donut17.png', 48)
    collection.addPieces('input/donut18.png', 48)
    collection.addPieces('input/donut19.png', 48)
    collection.addPieces('input/donut20.png', 9)
    collection.addPieces('input/donut21.png', 48)
    collection.addPieces('input/donut22.png', 29)
    collection.addPieces('input/donut23.png', 36)

    #collection.showPieceImages()
    #cv2.destroyAllWindows()

    print(len([piece for piece in collection.pieces if piece.type == 'corner']))
    print(len([piece for piece in collection.pieces if piece.type == 'side']))
    print(len([piece for piece in collection.pieces if piece.type == 'middle']))
    collection.showPieceImages()
    all_pieces = collection.getAllPiecesImage(with_details=True)
    cv2.imwrite('market_all_pieces.png', all_pieces)