import numpy as np
from scipy.signal import find_peaks
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
        self.image = image
        self.contour = contour
        self.corners = findCorners(self.contour)
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

        # crop to the circle
        image_crop = image_piece_isolated[max(int(y) - int(r), 0):min(int(y) + int(r), h), 
                                        max(int(x) - int(r), 0):min(int(x) + int(r), w)]

        # if the intended edge will actually be on the bottom
        if( c2[0] < c1[0] or (c2[0] == c1[0] and c2[1] < c1[1])): # need to rotate more!
            angle = angle + 180 # flip 180 degrees
        final_image = imutils.rotate(image_crop, angle) # rotate the image

        return final_image


def findCorners(contour):
    corners = np.empty((4,3), dtype=np.int)

    center = np.mean(contour[:,0], axis=0)
    centered_contour = contour - center

    rho, phi = cart2pol(centered_contour[:,0,0], centered_contour[:,0,1])
    peaks, _ = find_peaks(rho, distance=50)

    # delete potential duplicate peaks on the extreme ends (as 360 degrees is equal to 0 degrees)
    if (peaks[0] + len(rho) - peaks[-1] <= 50):
        if rho[peaks[0]] < rho[peaks[-1]]:
            peaks = peaks[1:]
        else:
            peaks = peaks[:-1]
    
    # peak with most extreme value
    rho_max = np.max(rho[peaks])

    # find the differences between the peak height and the points on either side
    # normalize so that the peak height is treated to be the same as rho_max
    # remove peaks that are not higher than the points on the left and right
    diff_left = rho_max - rho_max * rho[(peaks-30) % len(rho)] / rho[peaks]
    diff_left[np.where(diff_left < 0)] = 0
    diff_right = rho_max - rho_max * rho[(peaks+30) % len(rho)] / rho[peaks]
    diff_right[np.where(diff_right < 0)] = 0

    # find a metric for sharpness by multiplying these values
    sharpness = diff_left * diff_right

    # find the indeces of the four sharpest peaks, treat as corners
    corner_peaks = np.argsort(sharpness)[-4:]

    # reduce the peaks to be just the corners
    sharpness = sharpness[corner_peaks]
    peaks = peaks[corner_peaks]

    # sort in increasing order based on phi (clockwise)
    order = np.argsort([phi[peaks]])
    peaks = peaks[order][0]
    
    # find the coordinates of the corners
    corners_x = contour[:,0,0][peaks]
    corners_y = contour[:,0,1][peaks]
    
    # store in an array, return
    corners[:,0] = peaks
    corners[:,1] = corners_x
    corners[:,2] = corners_y

    return corners

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
        edges.append(new_edge)
    
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