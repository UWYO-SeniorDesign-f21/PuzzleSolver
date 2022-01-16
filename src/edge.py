import numpy as np
import cv2

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
    def __init__(self, number, image, contour):
        self.number = number
        self.image = image
        self.contour = contour
        self.distance_arr = findDistanceArray(self.contour)
        self.label = findLabel(self.distance_arr)
        self.color_arr = findColorArray(self.contour, self.image)
        self.left_neighbor = None
        self.right_neighbor = None

    def setLeftNeighbor(self, neighbor):
        self.left_neighbor = neighbor
    
    def setRightNeighbor(self, neighbor):
        self.right_neighbor = neighbor

    def compare(self, other_edge):
        if other_edge == self:
            #print(f'{self}, {other_edge}')
            return float('inf')
        # if there is a flat side, these should not go together
        if self.label == 'flat' or other_edge.label == 'flat':
            return float('inf')

        # check if the nieghbors of the edge are compatible
        if not self.right_neighbor or not self.left_neighbor:
            print('>:(')
        else:
            if (self.left_neighbor.label == 'flat') != (other_edge.right_neighbor.label == 'flat'):
                return float('inf')
            if (self.right_neighbor.label == 'flat') != (other_edge.left_neighbor.label == 'flat'):
                return float('inf')

            
        if self.label == other_edge.label:
            return float('inf')

        # find the shorter and the longer edge
        if len(self.contour) < len(other_edge.contour):
            short_edge, long_edge = self, other_edge
        else:
            short_edge, long_edge = other_edge, self

        # when puzzle edges are compared, they are flipped to be put together
        short_dists = -short_edge.distance_arr
        short_colors = short_edge.color_arr
        long_dists = np.flip(long_edge.distance_arr)
        long_colors = np.flip(long_edge.color_arr, axis=0)

        short_len = len(short_edge.contour)
        long_len = len(long_edge.contour)
        
        # for each possible position to compare at, find the distance
        # the closest the two pieces get to eachother is what should be compared
        min_diff = float('inf')
        min_diff_pos = 0
        for start_pos in range(len(long_dists) - len(short_dists) + 1):
            end_pos = start_pos + len(short_dists)
            # the difference between the two edges is the sum of differences at each point
            diff = np.sum(abs(short_dists - long_dists[start_pos:end_pos]))
            if diff < min_diff:
                min_diff = diff
                min_diff_pos = start_pos
        
        color_diff = np.sum(np.linalg.norm(long_colors[
                        max(min_diff_pos,1)-1:len(short_edge.color_arr)+max(min_diff_pos,1)-1] - short_edge.color_arr))
        return color_diff + min_diff + 2*(long_len - short_len)**2

def findDistanceArray(contour):
    if len(contour) == 0:
        return np.array([])
    c1 = contour[0] # find coords of first corner
    c2 = contour[-1] # find coords of second corner
    # for each point along the contour, find distance to line btw c1, c2
    return -np.cross(c2-c1,contour-c1)/np.linalg.norm(c2-c1)


def findLabel(distance_arr):
    if len(distance_arr) == 0:
        return 'none'
    # epsilon determines the distance from the line between corners to use as a threshold for 'flat'
    epsilon = 30
    extreme = np.max(abs(distance_arr)) # furthest point from the line
    if extreme <= epsilon: # below threshold
        label = 'flat'
    elif extreme == np.max(distance_arr): # extreme value goes away from center
        label = 'outer'
    else: # extreme value goes towards center
        label = 'inner'
    return label

def findColorArray(contour, image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels_in = range(4,5) # range of pixels from edge to average over

    # slope of perpendicular lines at each point on the contour
    perp_slopes = np.empty_like(contour[1:-1,0])

    # use the current, next, and previous points to find slope of line perpindicular at each point
    for i, curr_point in enumerate(contour[1:-1,0]):
        prev_point = contour[i][0]
        next_point = contour[i+2][0]
        # def of line perpinducular to another
        dx = -next_point[0] + prev_point[0]
        dy = -next_point[1] + prev_point[1]
        perp_slopes[i] = [dy, -dx]

    # ensure each slope has magnitude 1
    perp_slopes = perp_slopes / np.linalg.norm(perp_slopes, axis=1)[:,np.newaxis]
    # init color_arr
    color_arr = np.zeros((len(perp_slopes), 3))
    # for each slope, average the colors from each point along the slope
    for i, slope in enumerate(perp_slopes):
        color_sum = np.zeros((3,), dtype=int)
        for j in pixels_in:
            curr_point = contour[i+1, 0]
            sample_point = (curr_point + j*slope).astype(int)
            color_sum += image_gr[curr_point[1], curr_point[0]]
        color_arr[i] = color_sum // len(pixels_in)
    return color_arr