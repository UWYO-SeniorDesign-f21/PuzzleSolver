import numpy as np
import cv2
import math
import gc
from scipy import interpolate


'''
The edge class contains information about an edge of a puzzle piece.
Takes a contour, which is the part of the piece's contour along the edge, from corner to corner
The image is the full image of pieces that the edge is in
Finds the label, flat, outer, or inner
Finds the distances from the line between corners on a number of points along the contour, equally spaced
Finds the colors along the edge.
'''
class Edge:
    def __init__(self, number, contour, settings):
        self.points_per_side = settings[2]
        self.number = number
        self.corner_dist = np.linalg.norm(contour[0] - contour[-1]) # length of line btw corners
        self.contour, pts = self.getEquidistantPoints(contour) # space points equally along contour
        self.distance_arr = self.findDistanceArray(contour, pts) # get dist from line btw corners for each pt
        self.label = self.findLabel(self.distance_arr, contour) # flat, outer, inner
        self.color_arr = None # colors along contour, defined in piece
        self.left_neighbor = None # edge to the left on piece
        self.right_neighbor = None # edge to the right on piece
        self.color_histograms = None # histograms along the contour edge
        self.weights = None # used in calculating dists if not stored elsewhere
        self.mins = None
        self.maxs = None

    '''
    compares edges by calculating a score given weights, mins, and maxs using
    min-max normalization, weights, mins, maxs calculated in puzzleSolver in the function 
    getDistDict
    '''
    def compareWeighted(self, other):
        # if not a valid edge combo, return float('inf')
        if self.weights is None:
            return float('inf')
        entry = self.compare(other)
        if entry is None:
            return float('inf')
        dist_diff, color_diff, color_diff_hist, corner_diff = entry
        dist_diff = (dist_diff - self.mins[0]) / (self.maxs[0] - self.mins[0])
        color_diff = (color_diff - self.mins[1]) / (self.maxs[1] - self.mins[1])
        color_diff_hist = (color_diff_hist - self.mins[2]) / (self.maxs[2] - self.mins[2])
        corner_diff = (corner_diff - self.mins[3]) / (self.maxs[3] - self.mins[3])

        dist = self.weights[0]*dist_diff + self.weights[1]*color_diff + self.weights[2]*color_diff_hist + self.weights[3]*corner_diff
        return dist


    def setLeftNeighbor(self, neighbor):
        self.left_neighbor = neighbor
    
    def setRightNeighbor(self, neighbor):
        self.right_neighbor = neighbor

    def setColorHistogram(self, hist):
        self.color_histogram = hist

    '''
    compares self to other_edge, using the four metrics stored in each edge
    returns a value for each of these metrics
    used in the getDistDict function from puzzleSolver, where these scores are normalized
    '''
    def compare(self, other_edge):
        # if not a valid edge combo, return None
        if other_edge == self:
            return None
        if self.label == 'flat' or other_edge.label == 'flat':
            return None
        if (self.left_neighbor.label == 'flat') != (other_edge.right_neighbor.label == 'flat'):
            return None
        if (self.right_neighbor.label == 'flat') != (other_edge.left_neighbor.label == 'flat'):
            return None
        
        # mirror and invert metrics so that they are compared properly
        dist_arr_1 = -self.distance_arr
        dist_arr_2 = np.flip(other_edge.distance_arr)
        color_arr_2 = np.flip(other_edge.color_arr, axis=0)
        color_hists_2 = np.flip(other_edge.color_hists, axis=0)
        # l2 norm of difference of dist arrays
        dist_diff = math.sqrt(np.sum((dist_arr_1 - dist_arr_2)**2))
        # l2 norm of color differences
        color_diff = np.sum(math.sqrt(np.sum((self.color_arr - color_arr_2)**2)))
        # l2 norm of color histogram correlations
        color_diff_2 = 0
        for i, hist1 in enumerate(self.color_hists):
            hist2 = color_hists_2[i]
            color_diff_2 += (1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))**2
        color_diff_2 = math.sqrt(color_diff_2)
        # difference in length btw corners
        corner_dist_diff = abs(self.corner_dist - other_edge.corner_dist)
        corner_dist_ratio = max(self.corner_dist, other_edge.corner_dist) / min(self.corner_dist, other_edge.corner_dist)
        return dist_diff, color_diff, color_diff_2, corner_dist_ratio

    '''
    calculates the distance to the line between corners
    for each of the num_points points along the contour
    '''
    def findDistanceArray(self, contour, points):
        if len(contour) == 0:
            return np.array([])
        c1 = contour[0] # find coords of first corner
        c2 = contour[-1] # find coords of second corner
        # distance for each of the points on the contour
        d_arr = -np.cross(c2-c1,contour-c1)/np.linalg.norm(c2-c1)
        distance_arr_pts = [0]
        # smooth out the distance values
        for i, p1 in enumerate(points[:-2]):
            p3 = points[i+2]
            if p3 - (p1 + 1) <= 0:
                distance_arr_pts.append(d_arr[p1][0])
                continue
            avg_dist = np.mean(d_arr[p1+1:p3])
            distance_arr_pts.append(avg_dist)
        distance_arr_pts.append(0)
        return np.array(distance_arr_pts)

    '''
    finds the label for the edge, will be flat, inner, or outer
    '''
    def findLabel(self, distance_arr, contour):
        if len(distance_arr) == 0:
            return 'none'
        # epsilon determines the distance from the line between corners to use as a threshold for 'flat'
        epsilon = 10
        extreme = np.max(abs(distance_arr)) # furthest point from the line
        # extreme value of middle 50%, used to detect side edges, so that the corners don't mess w it
        middle_points = contour[math.floor(len(contour) / 8):math.ceil(len(contour) * 7 / 8)]
        middle_dists = self.findDistanceArray(middle_points, np.linspace(0, len(middle_points) - 1, num=self.points_per_side).astype(int))
        middle_extreme = np.max(abs(middle_dists))
        if middle_extreme <= epsilon: # below threshold
            label = 'flat'
        elif extreme == np.max(distance_arr): # extreme value goes away from center
            label = 'outer'
        else: # extreme value goes towards center
            label = 'inner'
        return label


    '''
    finds equally spaced points along the contour
    '''
    def getEquidistantPoints(self, contour):
        # length from beginnning to each point
        dists = [cv2.arcLength(contour[:i], False) for i in range(len(contour))]
        total_len = dists[-1]
        # desired distance from init point in contour, equally spaced
        desired_distances = np.linspace(0, total_len, num=self.points_per_side)

        # interpolate >:(

        interp = interpolate.interp1d(dists, range(len(contour)), kind="linear")
        pts = interp(desired_distances).astype(int)

        interp_cont_x = interpolate.interp1d(dists, contour[:,0,0], kind="linear")
        interp_cont_y = interpolate.interp1d(dists, contour[:,0,1], kind="linear")

        new_x = interp_cont_x(desired_distances)
        new_y = interp_cont_y(desired_distances)
        new_contour = None
        prev_point = None
        for i, x in enumerate(new_x):
            y = new_y[i]
            if new_contour is None:
                new_contour = np.array([[x, y]])
            else:
                new_contour = np.concatenate((new_contour, [[x, y]]))
        new_contour = new_contour[:,np.newaxis,:].astype(int)
        contour = np.unique(contour, axis=1)
        self.points_per_side = len(pts)
        return new_contour, pts