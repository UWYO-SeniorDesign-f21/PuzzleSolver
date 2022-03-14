import numpy as np
import cv2
import math
import gc

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
    def __init__(self, number, contour, settings):
        self.points_per_side = min(settings[2], len(contour))
        self.number = number
        self.corner_dist = np.linalg.norm(contour[0] - contour[-1])
        self.contour, pts = self.getEquidistantPoints(contour)
        self.distance_arr = self.findDistanceArray(contour, pts)
        self.label = self.findLabel(self.distance_arr, contour)
        self.color_arr = None
        self.left_neighbor = None
        self.right_neighbor = None
        self.color_histograms = None

    def clear(self):
        del self.color_histograms
        del self.color_arr
        # del self.distance_arr
        # del self.contour
        # del self.corner_dist
        # del self.points_per_side
        gc.collect()

    def setLeftNeighbor(self, neighbor):
        self.left_neighbor = neighbor
    
    def setRightNeighbor(self, neighbor):
        self.right_neighbor = neighbor

    def setColorHistogram(self, hist):
        self.color_histogram = hist

    def compare(self, other_edge):
        if other_edge == self:
            return None
        if self.label == 'flat' or other_edge.label == 'flat':
            return None
        if (self.left_neighbor.label == 'flat') != (other_edge.right_neighbor.label == 'flat'):
            return None
        if (self.right_neighbor.label == 'flat') != (other_edge.left_neighbor.label == 'flat'):
            return None
        dist_arr_1 = -self.distance_arr
        dist_arr_2 = np.flip(other_edge.distance_arr)
        color_arr_2 = np.flip(other_edge.color_arr, axis=0)
        color_hists_2 = np.flip(other_edge.color_hists, axis=0)
        # color_arr_hsv_2 = np.flip(other_edge.color_arr_hsv, axis=0)
        dist_diff = math.sqrt(np.sum((dist_arr_1 - dist_arr_2)**2))
        color_diff = np.sum(math.sqrt(np.sum((self.color_arr - color_arr_2)**2)))
        #color_diff_hsv = np.sum(np.linalg.norm(self.color_arr_hsv - color_arr_hsv_2))
        # color_diff = cv2.compareHist(self.color_histogram, other_edge.color_histogram, cv2.HISTCMP_CORREL)
        color_diff_2 = 0
        for i, hist1 in enumerate(self.color_hists):
            hist2 = color_hists_2[i]
            color_diff_2 += (1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))**2
        color_diff_2 = color_diff_2
        corner_dist_diff = abs(self.corner_dist - other_edge.corner_dist)
        corner_dist_ratio = max(self.corner_dist, other_edge.corner_dist) / min(self.corner_dist, other_edge.corner_dist)
        # print(color_diff, color_diff_old)
        # dist = (dist_diff + 10*self.points_per_side*(1 - color_diff)) * corner_dist_ratio
        # dist = (dist_diff + color_diff) * corner_dist_ratio
        # print(dist_diff, color_diff, color_diff_2, corner_dist_diff)
        return dist_diff, color_diff, color_diff_2, corner_dist_ratio

    def findDistanceArray(self, contour, points):
        if len(contour) == 0:
            return np.array([])
        c1 = contour[0] # find coords of first corner
        c2 = contour[-1] # find coords of second corner
        d_arr = -np.cross(c2-c1,contour-c1)/np.linalg.norm(c2-c1)
        distance_arr_pts = [0]
        for i, p1 in enumerate(points[:-2]):
            p3 = points[i+2]
            if p3 - (p1 + 1) <= 0:
                distance_arr_pts.append(d_arr[p1][0])
                continue
            avg_dist = np.mean(d_arr[p1+1:p3])
            distance_arr_pts.append(avg_dist)
        distance_arr_pts.append(0)
        return np.array(distance_arr_pts)

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

    def getEquidistantPoints(self, contour):
        dists = [cv2.arcLength(contour[:i], False) for i in range(len(contour))]
        total_len = dists[-1]
        desired_distances = np.linspace(0, total_len, num=self.points_per_side)

        import matplotlib.pyplot as plt
        from scipy import interpolate

        interp = interpolate.interp1d(dists, range(len(contour)), kind="linear")
        pts = interp(desired_distances).astype(int)
        pts = np.unique(pts)

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