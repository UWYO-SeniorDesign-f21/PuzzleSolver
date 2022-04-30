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
    def __init__(self, label, number, image, contour, settings):
        self.label = label # distinct label for piece
        self.number = number # index of piece in the collection's piece list
        self.image = image # image containing the piece
        self.contour = contour # contour along the edge of the piece
        self.findCorners() # find the corner locations for the piece
        self.settings = settings
        self.findEdges() # find the edges of the piece, as Edge objects
        self.getEdgeColors() # find colors for each edge, store in the Edge objects
        self.findType() # side, middle, or corner

    '''
    finds a cropped and rotated image for the piece, using the image the piece is in
    can be resized
    also returns the locations of the corners in this subimage
    used to put together the full image using the functions in PuzzleSolution
    '''
    def getSubimage2(self, edge_up, with_details=False, resize_factor=1, draw_edges=[], rel_edge=0, line_width=0):
        image = self.image.copy()
        h, w, _ = image.shape

        # used in demo to show edges on the piece
        for edge in range(4):
            if edge in draw_edges and len(draw_edges) == 4:
                cv2.drawContours(image, self.edges[edge].contour, -1, (0,255,0), thickness=10)

        cv2.drawContours(image, self.contour, -1, (0,0,0), thickness=line_width)

        # find a circle that encloses the piece in the image
        (x,y), r = cv2.minEnclosingCircle(self.contour)
        x = int(x)
        y = int(y)
        r = int(r)

        # adjust the circle if it would go over bounds of image
        if y - r < 0:
            y = r
        if x - r < 0:
            x = r

        h, w, _ = self.image.shape
        
        lpad = rpad = tpad = bpad = 0
        if y - r < 0:
            tpad = -(y - r)
        if x - r < 0:
            lpad = -(x - r)

        # isolate the piece in the image
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [self.contour], -1, (255,255,255), thickness=-1)
        image_piece_isolated = cv2.bitwise_and(mask, image)

        # show details if specified
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
        image_crop = image_piece_isolated[max(y-r, 0):min(y+r, h),max(x-r, 0):min(x+r, w)]
        h1, w1, _ = image_crop.shape

        padded_image = np.zeros((2*r, 2*r, 3), dtype=np.uint8)
        padded_image[tpad:tpad+h1, lpad:lpad+w1] = image_crop

        ph, pw, _ = padded_image.shape

        # get the corners on either end of the piece to be displayed on top
        c1 = self.corners[(edge_up + rel_edge) % 4 - 1][1:]
        c2 = self.corners[(edge_up + rel_edge) % 4][1:]

        # get the slope of the line to be on top
        delta = c1 - c2
        dx, dy = delta[0], delta[1]

        # if the line is vertical, rotate 90 degrees
        if dx == 0:
            angle = 90
        else: # otherwise, rotate arctan (change in y / change in x) degrees
            angle = math.degrees(math.atan(dy / dx))

        # if the intended edge will actually be on the bottom
        if( c2[0] < c1[0] or (c2[0] == c1[0] and c2[1] < c1[1])): # need to rotate more!
            angle = angle + 180 # flip 180 degrees

        angle -= 90 * rel_edge

        final_image = imutils.rotate(padded_image, angle) # rotate the image
        # resize the image
        final_image = cv2.resize(final_image, (int(ph * resize_factor), int(pw * resize_factor)), interpolation=cv2.INTER_AREA)
        
        return final_image, self.getAdjustedCorners(edge_up, (x, y), r, angle, resize_factor)

    '''
    finds the location of the corners in the subimage, where the piece is bounded by a circle and 
    the image is rotated (angle) degrees. 
    '''
    def getAdjustedCorners(self, edge_up, center, radius, angle, resize_factor):
        new_corners = []
        for i in range(edge_up - 1, edge_up + 3):
            index = i % 4
            corner = self.corners[index]
            # for i, corner in enumerate(self.corners):
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
            new_corners.append((np.array([x_new, y_new]) * resize_factor).astype(int))
        return np.array(new_corners)

    '''
    Finds the corner locations on the piece
    '''
    def findCorners(self):
        # dist, prominence to use in finding the peaks
        dist = len(self.contour) // 64
        sharpdist = len(self.contour) // 64 # how far to look on each side to detect sharpness
        prominence = 0

        # center the contour
        center = np.mean(self.contour[:,0], axis=0)
        centered_contour = self.contour - center
        
        # convert to polar coordinates and smooth
        rho, phi = cart2pol(centered_contour[:,0,0], centered_contour[:,0,1])
        rho = running_average(rho, 15)
        rho2 = np.concatenate((rho, rho, rho))
        # find peaks in the distance from the center of the piece
        peaks, _ = find_peaks(rho2, distance=dist, prominence=prominence)
        peaks = peaks[peaks >= len(rho)]
        peaks = peaks[peaks < 2*len(rho)]
        peaks -= len(rho)
        
        # find the sharpness of each peak, defined by the change in derivative
        # between neighboring points
        p1s = self.contour[(peaks - sharpdist) % len(self.contour)]
        p2s = self.contour[peaks]
        p3s = self.contour[(peaks + sharpdist) % len(self.contour)]

        d1 = p2s - p1s
        d2 = p3s - p2s
        
        delta = d1[:,0] - d2[:,0]
        sharpness = np.linalg.norm(delta, axis=1)

        # remove peaks with negative sharpness
        peaks = peaks[np.where(sharpness > 0)]
        sharpness = sharpness[np.where(sharpness > 0)]
    
        # reduce the peaks to be just the corners
        peaks = self.pickBestPeaks( peaks, sharpness )

        # sort in increasing order based on phi (clockwise)
        order = np.argsort([phi[peaks]])
        peaks = peaks[order][0]

        corners = np.zeros((4,3), dtype=np.int)

        # find the coordinates of the corners
        corners_x = self.contour[:,0,0][peaks]
        corners_y = self.contour[:,0,1][peaks]
        
        # store in an array, return
        corners[:len(peaks),0] = peaks
        corners[:len(peaks),1] = corners_x
        corners[:len(peaks),2] = corners_y

        self.corners = corners

    '''
    takes a list of peaks and returns the peaks estimated to be corners
    uses heuristics of sharpness, area covered, and rectangulareness to determine corners
    i.e. corners have high sharpness, high rectangularness, and high area covered
    relative to other subsets of the peaks
    '''
    def pickBestPeaks( self, peaks, sharpness ):
        # crop the image for efficiency
        (x,y), r = cv2.minEnclosingCircle(self.contour)
        r = int(r)
        x = int(x)
        y = int(y)
        img3 = np.zeros((2*r, 2*r), dtype=np.uint8)
        adj_contour = self.contour - [x-r, y-r]
        cv2.drawContours(img3, [adj_contour], -1, 255, thickness=-1)
        maxScore = -1
        maxPeaks = [0, 1, 2, 3]

        # normalize sharpness
        sharpness = ((sharpness - np.min(sharpness)) / (np.max(sharpness) - np.min(sharpness)))

        # iterate over subsets
        for i in range(len(peaks)):
            peak1 = peaks[i]
            for j in range(i+1, len(peaks)):
                peak2 = peaks[j]
                for k in range(j+1, len(peaks)):
                    peak3 = peaks[k]
                    for l in range(k+1, len(peaks)):
                        peak4 = peaks[l]
                        img2 = np.zeros_like(img3)
                        point1 = [adj_contour[peak1][0][0],  adj_contour[peak1][0][1]]
                        point2 = [adj_contour[peak2][0][0],  adj_contour[peak2][0][1]]
                        point3 = [adj_contour[peak3][0][0],  adj_contour[peak3][0][1]]
                        point4 = [adj_contour[peak4][0][0],  adj_contour[peak4][0][1]]

                        points = np.array([point1, point2, point3, point4])
                        
                        # find the area covered by the set of peaks
                        cv2.fillPoly(img2, pts=[points], color =255)
                        img4 = cv2.bitwise_and(img2, img3)

                        covered_area = np.sum(img4 == 255)

                        # find the rectangle that bounds the peaks
                        x,y,w,h = cv2.boundingRect(points)
                        rect = ((x, y), (w, h), 0)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)

                        # find the area covered by the box, and the area of the points
                        area_rect = cv2.contourArea(box)
                        area_points = cv2.contourArea(np.array([adj_contour[peak1], adj_contour[peak2], adj_contour[peak3], adj_contour[peak4]]))

                        # maximum when points form perfect rectangle
                        if area_rect > 0:
                            score_rect = (area_points / area_rect) ** (1/2)
                        else:
                            score_rect = 0

                        # maximum when all points are maximum sharpness
                        score_sharp = ((sharpness[i] + sharpness[j] + sharpness[k] + sharpness[l]) - max(sharpness[i], sharpness[j], sharpness[k], sharpness[l]))**(1/4)
                        
                        # combine metrics
                        score = (covered_area)*score_rect*score_sharp

                        if score > maxScore:
                            maxPeaks = [peak1, peak2, peak3, peak4]
                            maxScore = score

        return np.array(maxPeaks)

    '''
    Creates Edge objects based on the corners
    '''
    def findEdges(self):
        # init edges to empty
        edges = []

        # for each set of corners next to eachother
        for i in range(len(self.corners)):
            # get the corner positions in the contour
            c1_pos = self.corners[i-1][0]
            c2_pos = self.corners[i][0]
            # add a new edge which contains the contour between these two positions
            if c2_pos < c1_pos:
                new_edge = Edge(i, np.concatenate((self.contour[c1_pos:], self.contour[:c2_pos])), self.settings)
            else:
                new_edge = Edge(i, self.contour[c1_pos:c2_pos], self.settings)
            if len(edges) > 0:
                prev_edge = edges[-1]
                prev_edge.setRightNeighbor(new_edge)
                new_edge.setLeftNeighbor(prev_edge)
            edges.append(new_edge)

        edges[0].setLeftNeighbor(edges[-1])
        edges[-1].setRightNeighbor(edges[0])

        self.edges = edges

    '''
    Gets the average colors and color histograms along the contour
    where the edge is
    '''
    def getEdgeColors(self):
        # crop for efficiency
        (x,y), r = cv2.minEnclosingCircle(self.contour)
        r = int(r)
        x = int(x)
        y = int(y)

        h, w, _ = self.image.shape
        
        lpad = rpad = tpad = bpad = 0
        if y - r < 0:
            tpad = -(y - r)
        if x - r < 0:
            lpad = -(x - r)

        # isolate the piece in the image
        mask = np.zeros_like(self.image)
        cv2.drawContours(mask, [self.contour], -1, (255,255,255), thickness=-1)
        image_piece_isolated = cv2.bitwise_and(mask, self.image)

        # crop to the circle
        image_crop = image_piece_isolated[max(y-r, 0):min(y+r, h),max(x-r, 0):min(x+r, w)]
        h1, w1, _ = image_crop.shape

        padded_image = np.zeros((2*r, 2*r, 3), dtype=np.uint8)
        padded_image[tpad:tpad+h1, lpad:lpad+w1] = cv2.cvtColor(image_crop, cv2.COLOR_BGR2LAB)
        padded_image[:,:,0] = padded_image[:,:,0] // 2

        # erode the mask in order to increase accuracy
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

        adj_contour = self.contour - [x-r+lpad, y-r+tpad]
        piece_mask = np.zeros((2*r, 2*r), dtype=np.uint8)

        cv2.drawContours(piece_mask, [adj_contour], -1, 255, thickness=-1)
        piece_mask = cv2.erode(piece_mask, kernel, iterations=1)

        # for showing the colors as in the presentation
        piece_mask_show = np.zeros_like(padded_image)
        color_images = []

        # iterate over edges
        for i, edge in enumerate(self.edges):
            # adjust the contour to fit within the cropped image
            adj_contour = edge.contour - [x-r+lpad, y-r+tpad]
            edge_mask = np.zeros((2*r, 2*r), dtype=np.uint8)
            img_show = np.zeros((2*r, 2*r, 3), dtype=np.uint8)
            edge_colors = None
            edge_color_hists = []

            # find indices to look for a contour in
            hist_mask = np.zeros_like(edge_mask)
            hist_indices = np.linspace(2, edge.points_per_side - 3, max(4, edge.points_per_side // 8)).astype(int)[1:]
            
            starting_points = []
            corresponding_points = []
            # find points to look for color in
            for j in range(0, edge.points_per_side-1):
                if j == edge.points_per_side - 1:
                    p1 = adj_contour[j][0]
                    p2 = adj_contour[j-1][0]
                    dx, dy = p1 - p2
                else:
                    p1 = adj_contour[j][0]
                    p2 = adj_contour[j+1][0]
                    dx, dy = p2 - p1

                if dx == 0 and dy == 0:
                    if j > 0:
                        starting_points.append(starting_points[-1])
                        corresponding_points.append(corresponding_points[-1])
                        continue
                    else:
                        dx, dy = adj_contour[j+2][0] - p1

                perp_vector = ((1/np.linalg.norm([-dy, dx]))*np.array([-dy, dx]))
                
                starting_points.append((p1 + self.settings[0]*perp_vector).astype(int))
                corresponding_points.append((p1 + (self.settings[1] - self.settings[0])*perp_vector).astype(int))

            # iterate over points to look for colors in
            for j in range(2, edge.points_per_side-2):
                p1 = starting_points[j-1]
                p2 = starting_points[j+1]
                p3 = corresponding_points[j-1]
                p4 = corresponding_points[j+1]

                # get the color average
                edge_mask_2 = np.zeros((2*r, 2*r), dtype=np.uint8)
                cv2.drawContours(edge_mask_2, [np.array([p1, p3, p4, p2])], -1, 255, thickness=-1)

                edge_mask_2 = cv2.bitwise_and(edge_mask_2, piece_mask)
                hist_mask = cv2.bitwise_or(hist_mask, edge_mask_2)

                color_avg = cv2.mean(padded_image, mask=edge_mask_2)
                color = np.array([color_avg[0], color_avg[1], color_avg[2]])

                # # uncomment to make the gif
                # piece_mask_show[edge_mask_2 > 0] = padded_image[edge_mask_2 > 0]
                # cv2.imshow('em2', piece_mask_show)
                # if j % 4 == 0:
                #     color_images.append(piece_mask_show.copy())
                # cv2.waitKey(20)

                if edge_colors is None:
                    edge_colors = color
                else:
                    edge_colors = np.vstack((edge_colors, color))

                # calculate histogram
                if np.any(hist_indices == j):
                    hist = cv2.calcHist([padded_image], [0, 1, 2], hist_mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    edge_color_hists.append(hist)

                    hist_mask = np.zeros_like(hist_mask)


            edge.color_arr = edge_colors
            edge.color_hists = edge_color_hists

        # # uncomment to save the gif
        # for j in range(64):
        #     color_images.append(color_images[-1])
        # cv2.imshow('color_img', color_images[-1])
        # import imageio
        # imageio.mimsave(f'C:/Users/jimmy/Documents/SeniorDesign/PuzzleSolver/piece_gif.gif', color_images)
        # cv2.waitKey(0)

    '''
    Saves the type of the piece
    '''
    def findType(self):
        # get the number of flat edges on the piece
        num_flat = 0
        for edge in self.edges:
            if edge.label == 'flat':
                num_flat += 1

        # label appropriately
        if num_flat == 0:
            piece_type = 'middle'
        elif num_flat == 1:
            piece_type = 'side'
        else:
            piece_type = 'corner'

        self.type = piece_type

# convert from cartesian to polar coordinates, used in findCorners
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def running_average(x, n):
    from scipy.ndimage.filters import uniform_filter1d
    return uniform_filter1d(x, size=n, mode='wrap')