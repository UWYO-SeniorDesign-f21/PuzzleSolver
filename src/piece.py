import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import math
import imutils
import cv2

from edge import Edge

'''
The piece class contains all information about puzzle pieces.
The image is the image containing the piece in it,
the contour is the location of all the points around the edge of the piece
the label is a value used to describe the piece
the corners, edge objects, and type of the piece are all found.

The subimage containing the piece, cropped and rotated appropriately can be found with the function
getSubimage
'''
class Piece:
    def __init__(self, label, image, contour):
        self.label = label
        # print(label)
        self.image = image
        self.contour = contour
        self.corners = findCorners(self.contour, image)
        self.edges = findEdges(self.contour, self.corners, self.image)
        self.type = findType(self.edges)

    def getSubimage(self, edge_up, with_details=False):
        image = self.image.copy()
        h, w, _ = image.shape

        # get the corners on either end of the piece to be displayed on top
        c1 = self.corners[edge_up - 1][1:]
        c2 = self.corners[edge_up][1:]

        # get the slope of the line to be on top
        delta = c1 - c2
        dx, dy = delta[0], delta[1]

        # if the line is vertical, rotate 90 degrees
        if dx == 0:
            angle = 90
        else: # otherwise, rotate arctan (change in y / change in x) degrees
            angle = math.degrees(math.atan(dy / dx))

        # find a circle that encloses the piece in the image
        (x,y), r = cv2.minEnclosingCircle(self.contour)
        r = r + 10

        # isolate the piece in the image
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [self.contour], -1, (255,255,255), thickness=-1)
        image_piece_isolated = cv2.bitwise_and(mask, image)

        if with_details:
            for i, corner in enumerate(self.corners):
                prev = self.corners[i-1]
                cv2.circle(image_piece_isolated, (int(corner[1]), int(corner[2])), 10, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.line(image_piece_isolated, (int(corner[1]), int(corner[2])), (int(prev[1]), int(prev[2])), 
                        (0,255,0), thickness=1)

            for i, edge in enumerate(self.edges):
                if edge.label == 'flat':
                    color = (255,0,255)
                elif edge.label == 'inner':
                    color = (255,0,0)
                else:
                    color = (0,0,255)

                cv2.drawContours(image_piece_isolated, edge.contour, -1, color, thickness=5)

                # for i, point in enumerate(edge.contour[:,0]):
                #     if i % 5 == 0 and i < len(edge.color_arr):
                #         cv2.circle(image_piece_isolated, (point[0], point[1]), 5, edge.color_arr[i], thickness=-1, lineType=cv2.FILLED)

        # crop to the circle
        image_crop = image_piece_isolated[max(int(y) - int(r), 0):min(int(y) + int(r), h), 
                                        max(int(x) - int(r), 0):min(int(x) + int(r), w)]

        # if the intended edge will actually be on the bottom
        if( c2[0] < c1[0] or (c2[0] == c1[0] and c2[1] < c1[1])): # need to rotate more!
            angle = angle + 180 # flip 180 degrees

        final_image = imutils.rotate(image_crop, angle) # rotate the image

        return final_image

    def getSubimage2(self, edge_up, with_details=False):

        # get the corners on either end of the piece to be displayed on top
        c1 = self.corners[edge_up - 1][1:]
        c2 = self.corners[edge_up][1:]

        # get the slope of the line to be on top
        delta = c1 - c2
        dx, dy = delta[0], delta[1]

        # if the line is vertical, rotate 90 degrees
        if dx == 0:
            angle = 90
        else: # otherwise, rotate arctan (change in y / change in x) degrees
            angle = math.degrees(math.atan(dy / dx))

        # find a circle that encloses the piece in the image
        (x,y), r = cv2.minEnclosingCircle(self.contour)
        r = r + 1
        image = self.image[int(y-r):int(y+r), int(x-r):int(x+r)]
        adj_contour = self.contour - [int(x-r), int(y-r)]

        # isolate the piece in the image
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [adj_contour], -1, (255,255,255), thickness=-1)
        image_piece_isolated = cv2.bitwise_and(mask, image)

        # if the intended edge will actually be on the bottom
        if( c2[0] < c1[0] or (c2[0] == c1[0] and c2[1] < c1[1])): # need to rotate more!
            angle = angle + 180 # flip 180 degrees

        final_image = imutils.rotate(image_piece_isolated, angle) # rotate the image

        return final_image, (x, y), r, angle

# self on the left, piece2 on the right ::: 
def combineSubimagesRight(piece1, edge_up_1, piece2, edge_up_2):
    piece1_subimage, c1, r1, a1 = piece1.getSubimage2(edge_up=(edge_up_1+1)%4)
    piece1_subimage = imutils.rotate(piece1_subimage, -90)
    a1 -= 90
    piece1_adj_corners = getAdjustedCorners(piece1, c1, r1, a1)
    h1, w1, _ = piece1_subimage.shape

    top_right_corner = None
    for x, y in piece1_adj_corners:
        if (x - r1) > 0 and (y - r1) < 0:
            top_right_corner = (x, y)
        cv2.circle(piece1_subimage, (x, y), 10, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

    piece2_subimage, c2, r2, a2 = piece2.getSubimage2(edge_up=(edge_up_2-1)%4)
    piece2_subimage = imutils.rotate(piece2_subimage, 90)
    a2 += 90
    piece2_adj_corners = getAdjustedCorners(piece2, c2, r2, a2)
    h2, w2, _ = piece2_subimage.shape

    top_left_corner = None
    for i, corner in enumerate(piece2_adj_corners):
        x, y = corner
        if (x - r2) < 0 and (y - r2) < 0:
            top_left_corner = (x, y)
        cv2.circle(piece2_subimage, (x, y), 10, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)

    # now just want the top corners to line up
    start_pos = top_right_corner[0] - top_left_corner[0]
    start_pos_y_p1 = 0
    start_pos_y_p2 = 0
    diff_y = top_right_corner[1] - top_left_corner[1]
    if diff_y < 0:
        start_pos_y_p1 = -diff_y
    else:
        start_pos_y_p2 = diff_y
    # start_pos_y = top_right_corner[1] - top_left_corner[1]

    combined_subimage = np.zeros((max(h1, h2)+100, w1+w2+100, 3), dtype=np.uint8)
    combined_subimage[start_pos_y_p1:start_pos_y_p1+h1, 0:w1] = piece1_subimage
    combined_subimage[start_pos_y_p2:start_pos_y_p2+h2, start_pos:start_pos+w2] = cv2.bitwise_xor(combined_subimage[start_pos_y_p2:start_pos_y_p2+h2, start_pos:start_pos+w2], piece2_subimage)

    return combined_subimage

def getAdjustedCorners(piece, center, radius, angle):
    new_corners = []
    for i, corner in enumerate(piece.corners):
        x_old = corner[1]
        y_old = corner[2]
        # adjust so pivot is at origin
        x_adj = x_old - center[0]
        y_adj = y_old - center[1]
        # now rotate about origin
        x_new = x_adj*math.cos(math.radians(-angle)) - y_adj*math.sin(math.radians(-angle))
        y_new = y_adj*math.cos(math.radians(-angle)) + x_adj*math.sin(math.radians(-angle))
        # now add the radius so the corner is at the correct position in the subimage
        x_new += radius
        y_new += radius
        new_corners.append((int(x_new), int(y_new)))
    return new_corners

def findCorners(contour, image):
    
    x1 = np.min(contour[:,:,0])
    x2 = np.max(contour[:,:,0])
    y1 = np.min(contour[:,:,1])
    y2 = np.max(contour[:,:,1])


    dist = 3
    sharpdist = len(contour) // 128 # two dots on each side
    prominence = 0

    center = np.mean(contour[:,0], axis=0)
    centered_contour = contour - center

    rho, phi = cart2pol(centered_contour[:,0,0], centered_contour[:,0,1])
    rho = np.concatenate((rho[-10:], rho))
    peaks, _ = find_peaks(rho, distance=dist, prominence=prominence)
    rho = rho[10:]
    peaks -= 10
    if np.any(peaks < 0):
        if not np.any(peaks > len(rho) - dist):
            peaks[np.where(peaks < 0)] += len(phi)
        else:
            peaks = peaks[np.where(peaks >= 0)]

    # delete potential duplicate peaks on the extreme ends (as 360 degrees is equal to 0 degrees)
    if (peaks[0] + len(rho) - peaks[-1] <= dist):
        if rho[peaks[0]] < rho[peaks[-1]]:
            peaks = peaks[1:]
        else:
            peaks = peaks[:-1]
    
    # peak with most extreme value
    rho_max = np.max(rho[peaks])

    # find the differences between the peak height and the points on either side
    # normalize so that the peak height is treated to be the same as rho_max
    # remove peaks that are not higher than the points on the left and right
    diff_left = rho_max - rho_max * rho[(peaks-sharpdist) % len(rho)] / rho[peaks]
    #diff_left[np.where(diff_left < 0)] = 0
    diff_right = rho_max - rho_max * rho[(peaks+sharpdist) % len(rho)] / rho[peaks]
    #diff_right[np.where(diff_right < 0)] = 0

    # find a metric for sharpness by multiplying these values
    sharpness = np.min(np.vstack((diff_left, diff_right)), axis=0)
    peaks = peaks[np.where(sharpness > 0)]
    sharpness = sharpness[np.where(sharpness > 0)]

    # image2 = np.copy(image)

    # for peak in peaks:
    #     corner = contour[peak][0]
    #     cv2.circle(image2, (int(corner[0]), int(corner[1])), 10, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    # cv2.imshow('best peaks', image2[y1-10:y2+10, x1-10:x2+10])
    # cv2.waitKey(1)

    # plt.plot(rho)
    # plt.plot(peaks, rho[peaks], 'x')
    # plt.show()
    # print(peaks, sharpness)
    # reduce the peaks to be just the corners
    peaks = pickBestPeaks( contour, peaks, image, sharpness )
    # print(peaks)
    # sort in increasing order based on phi (clockwise)
    order = np.argsort([phi[peaks]])
    peaks = peaks[order][0]

    # image2 = np.copy(image)

    # for peak in peaks:
    #     corner = contour[peak][0]
    #     cv2.circle(image2, (int(corner[0]), int(corner[1])), 10, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    # cv2.imshow('best peaks', image2[y1-10:y2+10, x1-10:x2+10])
    # cv2.waitKey()

    # plt.plot(rho)
    # plt.plot(peaks, rho[peaks], 'x')
    # plt.show()

    corners = np.zeros((4,3), dtype=np.int)

    # find the coordinates of the corners
    corners_x = contour[:,0,0][peaks]
    corners_y = contour[:,0,1][peaks]
    
    # store in an array, return
    corners[:len(peaks),0] = peaks
    corners[:len(peaks),1] = corners_x
    corners[:len(peaks),2] = corners_y

    return corners

def pickBestPeaks( contour, peaks, img, sharpness ):

    # want the collection of peaks that maximizes area within the contour covered.
    (x,y), r = cv2.minEnclosingCircle(contour)
    r = int(r)
    x = int(x)
    y = int(y)
    img3 = np.zeros((2*r, 2*r), dtype=np.uint8)
    adj_contour = contour - [x-r, y-r]
    cv2.drawContours(img3, [adj_contour], -1, 255, thickness=-1)
    maxScore = -1
    maxPeaks = [0, 1, 2, 3]
    for i in range(len(peaks)):
        peak1 = peaks[i]
        for j in range(i+1, len(peaks)):
            peak2 = peaks[j]
            for k in range(j+1, len(peaks)):
                peak3 = peaks[k]
                for l in range(k+1, len(peaks)):
                    peak4 = peaks[l]
                    img2 = np.zeros_like(img3)
                    point1 = [contour[peak1][0][0],  contour[peak1][0][1]]
                    point2 = [contour[peak2][0][0],  contour[peak2][0][1]]
                    point3 = [contour[peak3][0][0],  contour[peak3][0][1]]
                    point4 = [contour[peak4][0][0],  contour[peak4][0][1]]

                    points = np.array([point1, point2, point3, point4])
                    adj_points = points - [x-r, y-r]
                    cv2.fillPoly(img2, pts=[adj_points], color =255)
                    img4 = cv2.bitwise_and(img2, img3)

                    covered_area = np.sum(img4 == 255)

                    #img4 = cv2.bitwise_and(img2, img3)
                    #img5 = cv2.bitwise_and(img2, cv2.bitwise_not(img3))

                    #area = np.sum(img4 == 255)
                    #badArea = np.sum(img5 == 255)

                    # score = area - badArea

                    #img6 = np.zeros_like(img)

                    rect = cv2.minAreaRect(points)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    # cv2.fillPoly(img6, pts=[box], color=(255,255,255))
                    # cv2.imshow('bees', img7[y1:y2, x1:x2])
                    # cv2.waitKey(0)

                    area_rect = cv2.contourArea(box)
                    area_points = cv2.contourArea(np.array([contour[peak1], contour[peak2], contour[peak3], contour[peak4]]))


                    # maximum when points form perfect rectangle
                    if area_rect > 0:
                        score_rect = (covered_area / area_rect)
                    else:
                        score_rect = 0
                    #print(area_points, area_rect)
                    score_sharp = sharpness[i] + sharpness[j] + sharpness[k] + sharpness[l]
                    #print(score)
                    score = covered_area * math.sqrt(score_sharp)

                    if score > maxScore:
                        maxPeaks = [peak1, peak2, peak3, peak4]
                        maxScore = score

    return np.array(maxPeaks)

def findEdges(contour, corners, image):
    # init edges to empty
    edges = []

    # for each set of corners next to eachother
    for i in range(len(corners)):
        # get the corner positions in the contour
        c1_pos = corners[i-1][0]
        c2_pos = corners[i][0]
        # add a new edge which contains the contour between these two positions
        if c2_pos < c1_pos:
            new_edge = Edge(i, image, np.concatenate((contour[c1_pos:], contour[:c2_pos])))
        else:
            new_edge = Edge(0, image, contour[c1_pos:c2_pos])
        if len(edges) > 0:
            prev_edge = edges[-1]
            prev_edge.setRightNeighbor(new_edge)
            new_edge.setLeftNeighbor(prev_edge)
        edges.append(new_edge)

    edges[0].setLeftNeighbor(edges[-1])
    edges[-1].setRightNeighbor(edges[0])
    
    return edges

def findType(edges):
    # get the number of flat edges on the piece
    num_flat = 0
    for edge in edges:
        if edge.label == 'flat':
            num_flat += 1

    # label appropriately
    if num_flat == 0:
        piece_type = 'middle'
    elif num_flat == 1:
        piece_type = 'side'
    else:
        piece_type = 'corner'

    return piece_type

# convert from cartesian to polar coordinates, used in findCorners
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi