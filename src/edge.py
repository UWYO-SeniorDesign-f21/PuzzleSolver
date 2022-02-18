import numpy as np
import cv2
import math

'''
The edge class contains information about an edge of a puzzle piece.
Takes a contour, which is the part of the piece's contour along the edge, from corner to corner
The image is the full image of pieces that the edge is in
Finds the label, flat, outer, or inner
Finds the distances from the line between corners and each point on the contour
Finds the colors along the edge.

Has one non-helper function that compares edges
'''
class Edge:
    def __init__(self, number, image, contour, settings):
        self.points_per_side = settings[2]
        self.number = number
        self.image = image
        self.corner_dist = np.linalg.norm(contour[0] - contour[-1])
        pts = np.linspace(0, len(contour) - 1, num=self.points_per_side).astype(int)
        self.contour = contour[pts]
        self.distance_arr = self.findDistanceArray(self.contour)
        self.label = self.findLabel(self.distance_arr)
        self.color_arr = findColorArray(self.contour, self.image, settings[:2], color_mode=0)
        self.color_arr_hsv = findColorArray(self.contour, self.image, settings[:2], color_mode=1)
        self.left_neighbor = None
        self.right_neighbor = None

    def setLeftNeighbor(self, neighbor):
        self.left_neighbor = neighbor
    
    def setRightNeighbor(self, neighbor):
        self.right_neighbor = neighbor

    def compare(self, other_edge):
        if other_edge == self:
            return float('inf')
        if self.label == 'flat' or other_edge.label == 'flat':
            return float('inf')
        if (self.left_neighbor.label == 'flat') != (other_edge.right_neighbor.label == 'flat'):
            return float('inf')
        if (self.right_neighbor.label == 'flat') != (other_edge.left_neighbor.label == 'flat'):
            return float('inf')
        dist_arr_1 = -self.distance_arr
        dist_arr_2 = np.flip(other_edge.distance_arr)
        color_arr_2 = np.flip(other_edge.color_arr, axis=0)
        color_arr_hsv_2 = np.flip(other_edge.color_arr_hsv, axis=0)
        dist_diff = np.sum(abs(dist_arr_1 - dist_arr_2))
        color_diff = color_diff = np.sum(np.linalg.norm(self.color_arr - color_arr_2))
        color_diff_hsv = np.sum(np.linalg.norm(self.color_arr_hsv - color_arr_hsv_2))
        corner_dist_diff = abs(self.corner_dist - other_edge.corner_dist)
        corner_dist_ratio = max(self.corner_dist, other_edge.corner_dist) / min(self.corner_dist, other_edge.corner_dist)
        return (dist_diff + color_diff) * (corner_dist_ratio**2)

    def findDistanceArray(self, contour):
        if len(contour) == 0:
            return np.array([])
        c1 = contour[0] # find coords of first corner
        c2 = contour[-1] # find coords of second corner
        d_arr = -np.cross(c2-c1,contour-c1)/np.linalg.norm(c2-c1)
        return d_arr

    def findLabel(self, distance_arr):
        if len(distance_arr) == 0:
            return 'none'
        # epsilon determines the distance from the line between corners to use as a threshold for 'flat'
        epsilon = 10
        extreme = np.max(abs(distance_arr)) # furthest point from the line
        # extreme value of middle 50%, used to detect side edges, so that the corners don't mess w it
        middle_points = self.contour[math.floor(self.points_per_side / 4):math.ceil(self.points_per_side * 3 / 4)]
        middle_dists = self.findDistanceArray(middle_points)
        middle_extreme = np.max(abs(middle_dists))
        if middle_extreme <= epsilon: # below threshold
            label = 'flat'
        elif extreme == np.max(distance_arr): # extreme value goes away from center
            label = 'outer'
        else: # extreme value goes towards center
            label = 'inner'
        return label

def findColorArray(contour, image, settings, color_mode=0):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels_in = range(settings[0],settings[1]) # range of pixels from edge to average over

    # slope of perpendicular lines at each point on the contour
    perp_slopes = np.empty_like(contour[1:-1,0])

    # use the current, next, and previous points to find slope of line perpindicular at each point
    for i, curr_point in enumerate(contour[:,0]):
        if i < 1 or i > len(contour) - 2:
            continue
        prev_point = contour[i-1][0]
        next_point = contour[i+1][0]
        # def of line perpinducular to another
        dx = -next_point[0] + prev_point[0]
        dy = -next_point[1] + prev_point[1]
        perp_slopes[i-1] = [dy, -dx]
        if dy == 0 and dx == 0:
            perp_slopes[i-1] = perp_slopes[i-2]

    # ensure each slope has magnitude 1

    norm = np.linalg.norm(perp_slopes, axis=1)[:,np.newaxis]
    perp_slopes = perp_slopes / norm
    # init color_arr
    color_arr = np.zeros((len(perp_slopes), 3))
    # for each slope, average the colors from each point along the slope
    for i, slope in enumerate(perp_slopes):
        color_sum = np.zeros((3,), dtype=float)
        for j in pixels_in:
            curr_point = contour[i+1, 0]
            sample_point = (curr_point + j*slope).astype(int)
            if color_mode == 0:
                color_sum += image[sample_point[1], sample_point[0]]
            elif color_mode == 1:
                color_sum += np.dot(image_hsv[sample_point[1], sample_point[0]], [255/179, 1, 1])
            else:
                color_sum += image_gr[sample_point[1], sample_point[0]]
            # image2 = image.copy()
            # cv2.circle(image2, (sample_point[0], sample_point[1]), 2, (0,255,255), thickness=-1, lineType=cv2.FILLED)
            # cv2.imshow('image', cv2.resize(image2[sample_point[1]-30:sample_point[1]+30, sample_point[0]-30:sample_point[0]+30], (200,200), interpolation=cv2.INTER_AREA))
            # cv2.waitKey(1)
        color_arr[i] = color_sum / len(pixels_in)
    return color_arr