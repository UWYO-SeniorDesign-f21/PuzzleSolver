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
        points_per_side = 32
        self.number = number
        self.image = image
        self.corner_dist = np.linalg.norm(contour[0] - contour[-1])
        pts = np.linspace(0, len(contour) - 1, num=points_per_side).astype(int)
        self.contour = contour[pts]
        self.distance_arr = findDistanceArray(self.contour, self.image)
        self.label = findLabel(self.distance_arr)
        #self.distance_arr = smoothDistanceArray(self.distance_arr, 15)
        self.color_arr = findColorArray(self.contour, self.image, color_mode=0)
        self.color_arr_hsv = findColorArray(self.contour, self.image, color_mode=1)
        #self.color_arr = smoothColorArray(self.color_arr, 15)
        #self.color_arr_hsv = smoothColorArray(self.color_arr_hsv, 15)
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
        #print(dist_diff, color_diff, color_diff_hsv, corner_dist_ratio**2)
        return (dist_diff + color_diff) * (corner_dist_ratio**2)
        # pad_width = 20 # wiggle room
        # if other_edge == self:
        #     #print(f'{self}, {other_edge}')
        #     return float('inf')
        # # if there is a flat side, these should not go together
        # if self.label == 'flat' or other_edge.label == 'flat':
        #     return float('inf')

        # # check if the nieghbors of the edge are compatible
        # if not self.right_neighbor or not self.left_neighbor:
        #     print('>:(')
        # else:
        #     if (self.left_neighbor.label == 'flat') != (other_edge.right_neighbor.label == 'flat'):
        #         return float('inf')
        #     if (self.right_neighbor.label == 'flat') != (other_edge.left_neighbor.label == 'flat'):
        #         return float('inf')

        # # if self.label == other_edge.label:
        # #     return float('inf')

        # # find the shorter and the longer edge
        # if len(self.contour) < len(other_edge.contour):
        #     short_edge, long_edge = self, other_edge
        # else:
        #     short_edge, long_edge = other_edge, self

        # # when puzzle edges are compared, they are flipped to be put together
        # short_dists = -short_edge.distance_arr
        # short_colors = short_edge.color_arr
        # short_colors_hsv = short_edge.color_arr_hsv
        # long_dists = np.flip(long_edge.distance_arr)
        # long_colors = np.flip(long_edge.color_arr, axis=0)
        # long_colors_hsv = np.flip(long_edge.color_arr_hsv, axis=0)

        # short_len = len(short_edge.contour)
        # long_len = len(long_edge.contour)

        # compare_range_min = -pad_width
        # compare_range_max = long_len - short_len + 1 + pad_width
        
        # # for each possible position to compare at, find the distance
        # # the closest the two pieces get to eachother is what should be compared
        # min_diff = float('inf')
        # min_diff_pos = 0
        # min_pad = 0
        # min_bounds = None
        # for start_pos in range(compare_range_min, compare_range_max):
        #     long_start = max(start_pos, 0)
        #     long_end = min(start_pos + short_len, long_len)
        #     pad = 0
        #     if start_pos < 0:
        #         short_start = -start_pos
        #         short_start_color = short_start
        #         pad = -start_pos
        #     else:
        #         short_start = 0
        #         short_start_color = 0
        #     if start_pos >= long_len - short_len + 1:
        #         short_end = long_len - start_pos
        #         pad = short_len - short_end
        #     else:
        #         short_end = short_len
        #     # short_end_color = short_end - 1
        #     # short_start_color = short_start + 1
        #     # if short_end_color == short_len - 1:
        #     #     short_end_color -= 1
        #     #     short_start_color -= 1
        #     # long_end_color = long_end - 1
        #     # long_start_color = long_start + 1
        #     # if long_end_color == long_len - 2:
        #     #     long_end_color -= 1
        #     #     long_start_color -= 1
        #     # if long_end_color > long_len - 2:
        #     #     long_start_color -= 1
        #     end_pos = start_pos + len(short_dists)
        #     # the difference between the two edges is the sum of differences at each point
        #     dist_diff = np.sum(abs(short_dists[short_start:short_end] - long_dists[long_start:long_end]))
        #     #color_diff = np.sum(np.linalg.norm(long_colors[long_start_color:long_end_color] - short_edge.color_arr[short_start_color:short_end_color]))
        #     #color_diff_hsv = np.sum(np.linalg.norm(long_colors_hsv[long_start_color:long_end_color] - short_edge.color_arr_hsv[short_start_color:short_end_color]))
        #     #color_diff_hsv = np.sum(np.linalg.norm(long_colors_hsv[start_pos:start_pos+len(short_edge.color_arr)] - short_edge.color_arr_hsv))
        #     #diff = dist_diff + color_diff + color_diff_hsv + (long_len - short_len)**2
        #     diff = dist_diff# + color_diff + color_diff_hsv
        #     avg_diff = diff / (short_end - short_start)
        #     diff += pad * avg_diff
        #     diff += (long_len - short_len)**2
        #     if diff < min_diff:
        #         min_pad = pad
        #         min_diff = diff
        #         min_diff_pos = start_pos
        #         min_bounds = (short_start, short_end, long_start, long_end)

        # short_start, short_end, long_start, long_end = min_bounds
        # short_end_color = short_end - 1
        # short_start_color = short_start + 1
        # if short_end_color == short_len - 1:
        #     short_end_color -= 1
        #     short_start_color -= 1
        # long_end_color = long_end - 1
        # long_start_color = long_start + 1
        # if long_end_color == long_len - 2:
        #     long_end_color -= 1
        #     long_start_color -= 1
        # if long_end_color > long_len - 2:
        #     long_start_color -= 1
        # color_diff = np.sum(np.linalg.norm(long_colors[long_start_color:long_end_color] - short_edge.color_arr[short_start_color:short_end_color]))
        # color_diff_hsv = np.sum(np.linalg.norm(long_colors_hsv[long_start_color:long_end_color] - short_edge.color_arr_hsv[short_start_color:short_end_color]))
        # min_diff += color_diff
        # min_diff += color_diff_hsv
        # return min_diff 

def findDistanceArray(contour, image):
    if len(contour) == 0:
        return np.array([])
    c1 = contour[0] # find coords of first corner
    c2 = contour[-1] # find coords of second corner

    d_arr = -np.cross(c2-c1,contour-c1)/np.linalg.norm(c2-c1)
    # colors = 50 * ((d_arr - np.min(d_arr)) / np.max(d_arr - np.min(d_arr)))
    # print(np.min(colors), np.max(colors))

    # min_x = min(new_contour[:,0], key=lambda x:x[0])[0] - 40
    # max_x = max(new_contour[:,0], key=lambda x:x[0])[0] + 40

    # min_y = min(new_contour[:,0], key=lambda x:x[1])[1] - 40
    # max_y = max(new_contour[:,0], key=lambda x:x[1])[1] + 40
    # image2 = image.copy()
    # for i, point in enumerate(new_contour):
    #     pt = point[0]
    #     x = pt[0]
    #     y = pt[1]
    #     color = np.uint8([[[colors[i],255,255]]])
    #     color_bgr = cv2.cvtColor(color,cv2.COLOR_HSV2BGR)[0][0]
    #     color_bgr = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
    #     print(color_bgr)
    #     cv2.circle(image2, (int(x), int(y)), 5, color_bgr, thickness=-1, lineType=cv2.FILLED)
    # cv2.imshow('image', image2[min_y:max_y, min_x:max_x])
    # cv2.waitKey()
    # for each point along the contour, find distance to line btw c1, c2
    return d_arr

def smoothDistanceArray(dist_arr, n):
    dist_arr = np.append(np.repeat(dist_arr[0], int(n/2)), dist_arr)
    dist_arr = np.append(dist_arr, np.repeat(dist_arr[-1], int(n/2)))
    ret = np.cumsum(dist_arr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def smoothColorArray(color_arr, n):
    color_arr = np.vstack((np.tile(color_arr[0], (int(n/2), 1)), color_arr))
    color_arr = np.vstack((color_arr, np.tile(color_arr[-1], (int(n/2), 1))))
    ret = np.cumsum(color_arr, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def findLabel(distance_arr):
    if len(distance_arr) == 0:
        return 'none'
    # epsilon determines the distance from the line between corners to use as a threshold for 'flat'
    epsilon = 20
    extreme = np.max(abs(distance_arr)) # furthest point from the line
    if extreme <= epsilon: # below threshold
        label = 'flat'
    elif extreme == np.max(distance_arr): # extreme value goes away from center
        label = 'outer'
    else: # extreme value goes towards center
        label = 'inner'
    return label

def findColorArray(contour, image, color_mode=0):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels_in = range(18,20) # range of pixels from edge to average over

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