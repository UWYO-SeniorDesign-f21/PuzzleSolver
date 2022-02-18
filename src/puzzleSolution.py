from pieceCollection import PieceCollection
import random
import cv2
import numpy as np
from timeit import default_timer as timer
import math
import copy

class PuzzleSolution:
    def __init__(self, pieces, dims, dist_dict, sorted_dists, empty_edge_dist, cutoff=float('inf')):
        self.pieces = pieces # pieceCollection object
        self.puzzle_dims = dims
        self.dist_dict = dist_dict # distance btw all edges
        self.sorted_dists = sorted_dists

        self.position_dict = {} # x y coords to piece
        self.left_edge = self.right_edge = self.bottom_edge = self.top_edge = None
        self.info_dict = {} # piece to x y coords, edge up

        self.edges = set()
        self.start = None

        self.score = 0
        self.similarity_score = 0
        self.empty_edge_dist = empty_edge_dist
        self.cutoff = cutoff

        self.all_piece_dims = [0,0,0,0]
        self.middle_piece_dims = None

    def crossover(self, other):
        common_edges = self.edges.intersection(other.edges)
        new_solution = PuzzleSolution(self.pieces, self.puzzle_dims, self.dist_dict, self.sorted_dists, self.empty_edge_dist, cutoff=self.cutoff)
        new_solution.solvePuzzle(random_start=True, include_edges=common_edges)
        return new_solution

    def mutate(self, mutation_rate):
        roll = random.uniform(0, 1)
        new_solution = PuzzleSolution(self.pieces, self.puzzle_dims, self.dist_dict, self.sorted_dists, self.empty_edge_dist, self.cutoff)

        new_solution.edges = self.edges.copy()
        new_solution.info_dict = self.info_dict.copy()
        new_solution.position_dict = self.position_dict.copy()
        new_solution.score = self.score

        if roll > mutation_rate:         
            return new_solution

        while roll <= mutation_rate:
            swap_cost = float('inf')
            swap = None
            # avoids bad swaps. these are rare, so probably will only iterate once
            while swap_cost == float('inf'):
                swap = new_solution.getSwap()
                # will either update self.edges or return cost inf
                swap_cost = new_solution.swapEdges(swap)

            # print(self.score, swap_cost)
            new_solution.score += swap_cost

            piece1, edge1, piece2, edge2 = swap

            piece1_pos_old, piece1_edge_up_old = new_solution.info_dict[piece1]
            piece2_pos_old, piece2_edge_up_old = new_solution.info_dict[piece2]
            diff_piece2 = piece2_edge_up_old - edge2
            diff_piece1 = piece1_edge_up_old - edge1
            piece1_edge_up_new = (edge1 + diff_piece2) % 4
            piece2_edge_up_new = (edge2 + diff_piece1) % 4
            new_solution.info_dict[piece1] = (piece2_pos_old, piece1_edge_up_new)
            new_solution.info_dict[piece2] = (piece1_pos_old, piece2_edge_up_new)
            new_solution.position_dict[piece2_pos_old] = (piece1, piece1_edge_up_new)
            new_solution.position_dict[piece1_pos_old] = (piece2, piece2_edge_up_new)

            roll = random.uniform(0, 1)

        return new_solution

    def getSwap(self):
        random_edges = random.sample(self.edges, k=min(len(self.edges), 32))
        piece1, edge1, piece3, edge3 = max(random_edges, key=lambda x:self.dist_dict[(x[0], x[1])][x[2], x[3]])

        sorted_dists = self.sorted_dists[(piece3, edge3)]              
        piece2 = None
        while piece2 is None:
            piece2_choices = random.sample(sorted_dists, k=min(len(sorted_dists), 32))
            piece2, edge2 = min(piece2_choices, key=lambda x:x[1])[0]
            if piece2 == piece1 and edge1 == edge2:
                piece2 = None
        swap = (piece1, edge1, piece2, edge2)
        return swap

    # swaps the edges and returns the net cost of the swap
    def swapEdges(self, swap):
        swap_cost = 0
        piece1_swap, edge1_swap, piece2_swap, edge2_swap = swap

        if piece1_swap == piece2_swap:
            return float('inf')

        used_edges = [edge for edge in self.edges if edge[0] == piece1_swap or edge[0] == piece2_swap]
        add_edges = set()
        remove_edges = set()
        for i in range(4):
            piece1_edge = (edge1_swap + i) % 4
            piece2_edge = (edge2_swap + i) % 4
            list_edges =  [edge for edge in used_edges if edge[0] == piece1_swap and edge[1] == piece1_edge]
            if len(list_edges) > 0:
                _, _, piece3, piece3_edge = list_edges[0]
            else:
                continue

            list_edges = [edge for edge in used_edges if edge[0] == piece2_swap and edge[1] == piece2_edge]
            if len(list_edges) > 0:
                _, _, piece4, piece4_edge = list_edges[0]
            else:
                continue

            # swap those edges >:(
            if not (piece1_swap, piece1_edge, piece3, piece3_edge) in remove_edges:
                remove_edges.add((piece1_swap, piece1_edge, piece3, piece3_edge))
                remove_edges.add((piece3, piece3_edge, piece1_swap, piece1_edge))
                swap_cost -= self.dist_dict[(piece1_swap, piece1_edge)][(piece3, piece3_edge)]

            if not (piece2_swap, piece2_edge, piece4, piece4_edge) in remove_edges:
                remove_edges.add((piece2_swap, piece2_edge, piece4, piece4_edge))
                remove_edges.add((piece4, piece4_edge, piece2_swap, piece2_edge))
                swap_cost -= self.dist_dict[(piece2_swap, piece2_edge)][(piece4, piece4_edge)]

            if not (piece1_swap, piece1_edge, piece4, piece4_edge) in add_edges:
                add_edges.add((piece1_swap, piece1_edge, piece4, piece4_edge))
                add_edges.add((piece4, piece4_edge, piece1_swap, piece1_edge))
                swap_cost += self.dist_dict[(piece1_swap, piece1_edge)][(piece4, piece4_edge)]

            if not (piece2_swap, piece2_edge, piece3, piece3_edge) in add_edges:
                add_edges.add((piece2_swap, piece2_edge, piece3, piece3_edge))
                add_edges.add((piece3, piece3_edge, piece2_swap, piece2_edge))
                swap_cost += self.dist_dict[(piece2_swap, piece2_edge)][(piece3, piece3_edge)]
        
        if swap_cost == 0 or swap_cost == float('inf'):
            return float('inf')
        for edge in remove_edges:
            self.edges.remove(edge)
        for edge in add_edges:
            self.edges.add(edge)
        return swap_cost

    def solvePuzzle(self, random_start=False, show_solve=False, start=None, include_edges=set()):
        if not random_start and start:
            self.start = start
            piece1 = start
            edge1 = random.choice(range(4))
        else:
            piece1 = random.choice(self.pieces.pieces)
            self.start = piece1
            edge1 = random.choice(range(4))
        
        # piece1 at position 0,0 with edge1 up
        self.position_dict[(0,0)] = (piece1, edge1)

        self.updateEdges(piece1, (0,0), edge1)

        # init helper sets
        remaining_pieces = set(self.pieces.pieces) # available pieces to use
        remaining_pieces.remove(piece1)

        self.info_dict[piece1] = ((0, 0), edge1)

        if piece1.type == 'middle':
            self.middle_piece_dims = [0,0,0,0]

        kernel = set() # group of possible edges to extend off of
        kernel_num_edges = {}
        for edge in range(4):
            kernel.add((piece1, edge))
            kernel_num_edges[(piece1, edge)] = 1

        if len(include_edges) > 0:
            edges_to_include = [edge for edge in include_edges if (edge[0], edge[1]) in kernel]
        else:
            edges_to_include = []

        # while pieces are able to be added
        while len(remaining_pieces) > 0 and len(kernel) > 0:
            # print(len(remaining_pieces), len(kernel))
            min_edge = None
            min_adj = None
            if len(edges_to_include) > 0:
                min_edge, min_adj = self.getNextEdgeCrossover(edges_to_include, kernel_num_edges)
            if min_edge is None:
                min_edge, min_adj = self.getNextEdge(kernel, kernel_num_edges, remaining_pieces)
            if min_edge is None: # if there are no possible piece locations
                break
            # get the closest edge
            piece1, edge1, piece2, edge2 = min_edge
            remaining_pieces.remove(piece2) # update available pieces, as piece2 is added to group

            # update the score, kernel
            for edge in min_adj:
                dist = self.dist_dict[edge[0], edge[1]][edge[2], edge[3]]
                self.score += dist
                if dist < self.cutoff:
                    self.edges.add(edge)
                    self.edges.add((edge[2], edge[3], edge[0], edge[1]))
                if (edge[0], edge[1]) in kernel:
                    kernel.remove((edge[0], edge[1]))
                    kernel_num_edges.pop((edge[0], edge[1]), None)
                else:
                    print('zoinks')

            # get the location and edge up of piece 2, update info dict, position dict
            piece2_loc, piece2_edge_up = getPieceInfo(piece1, edge1, edge2, self.info_dict)

            self.info_dict[piece2] = (piece2_loc, piece2_edge_up)
            self.position_dict[piece2_loc] = (piece2, piece2_edge_up)

            # add edges that are facing an empty spot to kernel
            used_edges = [edge[3] for edge in min_adj]
            for edge in range(4):
                if edge not in used_edges:
                    new_piece_loc, _ = getPieceInfo(piece2, edge, 0, self.info_dict)
                    if self.position_dict.get(new_piece_loc):
                        continue
                    elif piece2.edges[edge].label != 'flat':
                        kernel.add((piece2, edge))
                        num_adj = len(self.getAdjacentEdges(piece2, edge, None, 0))
                        kernel_num_edges[(piece2, edge)] = num_adj

            if len(include_edges) > 0:
                edges_to_include = []
                for edge in include_edges:
                    if (edge[0], edge[1]) in kernel and edge[2] in remaining_pieces:
                        edges_to_include.append(edge)

            # get the locations of the left, right, top, and bottom edges of the puzzle if applicable
            self.updateEdges(piece2, piece2_loc, piece2_edge_up)
            self.updatePieceDims(piece2, piece2_loc)
            if show_solve:
                print(len(remaining_pieces), len(kernel))
                print(self.all_piece_dims, self.middle_piece_dims)
                image = self.getSolutionImage(with_details=True)
                h, w, _  = image.shape
                cv2.imshow(f'solution', cv2.resize(image, (int(500 * (w / h)), 500), interpolation=cv2.INTER_AREA))
                cv2.waitKey()
        # for any empty spots, penalty
        self.score += 4*len(remaining_pieces)*self.empty_edge_dist

    def getNextEdge(self, kernel, kernel_num_edges, remaining_pieces):
        min_edge = None
        min_adj = None
        min_dist = float('inf')
        # min_buddy = None
        # min_buddy_adj = None
        # min_buddy_dist = float('inf')
        # loop over possible edges to extend and possible edges to connect to them
        for num_adj in [4, 3, 2, 1]:
            kernel_num_adj = [key for key in kernel if kernel_num_edges.get(key) == num_adj]
            for piece1, edge1 in kernel_num_adj:
                piece1_loc, piece1_edge_up = self.info_dict[piece1]
                for entry, dist in self.sorted_dists[(piece1, edge1)]:
                    piece2, edge2 = entry
                    if piece2 in remaining_pieces:
                #for piece2 in remaining_pieces:
                    #for edge2 in range(4):
                        # get the edges that are adjacent to the piece being inserted
                        adj_edges = self.getAdjacentEdges(piece1, edge1, piece2, edge2)
                        dist = 0
                        # check if each edge is valid, find the distance
                        valid = True
                        #buddies = True
                        for adj_edge in adj_edges:
                            adj_dist = self.dist_dict[(adj_edge[0], adj_edge[1])][(adj_edge[2], adj_edge[3])]
                            #if not adj_dist == min(self.dist_dict[(adj_edge[2], adj_edge[3])].values()) or not adj_dist == min(self.dist_dict[(adj_edge[0], adj_edge[1])].values()):
                            #    buddies = False
                            dist += adj_dist
                        if dist >= min_dist:
                            continue
                        for adj_edge in adj_edges:
                            if not self.isValid(adj_edge[0], adj_edge[1], adj_edge[2], adj_edge[3]):
                                valid = False
                                break   
                        if not valid:
                            continue                 
                        # if buddies and dist < min_buddy_dist:
                        #     min_buddy_dist = dist
                        #     min_buddy = (piece1, edge1, piece2, edge2)
                        #     min_buddy_adj = adj_edges

                        # update minimum distance for that number of adj edges
                        min_dist = dist
                        min_edge = (piece1, edge1, piece2, edge2)
                        min_adj = adj_edges
                        if num_adj == 1:
                            break
            if not min_edge is None:
                break
        #if min_buddy is None:
        #    return min_edge, min_adj
        #else:
        #    return min_buddy, min_buddy_adj
        return min_edge, min_adj

    def getNextEdgeCrossover(self, edges_to_include, kernel_num_edges):
        min_edge = None
        min_adj = None
        min_dist = float('inf')
        # min_buddy = None
        # min_buddy_adj = None
        # min_buddy_dist = float('inf')
        for num_adj in [4, 3, 2, 1]:
            edges_num_adj = [edge for edge in edges_to_include if kernel_num_edges.get((edge[0], edge[1])) == num_adj]
            # loop over possible edges to extend and possible edges to connect to them
            for edge in edges_num_adj:
                piece1, edge1, piece2, edge2 = edge
                piece1_loc, piece1_edge_up = self.info_dict[piece1]
                adj_edges = self.getAdjacentEdges(piece1, edge1, piece2, edge2)
                dist = 0
                # check if each edge is valid, find the distance
                #buddies = True
                for adj_edge in adj_edges:
                    # if not adj_edge in edges_to_include:
                    #     valid=False
                    #     break
                    adj_dist = self.dist_dict[(adj_edge[0], adj_edge[1])][(adj_edge[2], adj_edge[3])]
                    # if not adj_dist == min(self.dist_dict[(adj_edge[2], adj_edge[3])].values()) or not adj_dist == min(self.dist_dict[(adj_edge[0], adj_edge[1])].values()):
                    #     buddies = False
                    dist += adj_dist

                if dist == float('inf'):
                    break
                valid = True
                for adj_edge in adj_edges:
                    if not self.isValid(adj_edge[0], adj_edge[1], adj_edge[2], adj_edge[3]):
                        valid = False
                        break

                if not valid:
                    continue
                # if buddies and dist < min_buddy_dist:
                #     min_buddy_dist = dist
                #     min_buddy = (piece1, edge1, piece2, edge2)
                #     min_buddy_adj = adj_edges
                # update minimum distance for that number of adj edges
                min_dist = dist
                min_edge = (piece1, edge1, piece2, edge2)
                min_adj = adj_edges
            if not min_edge is None:
                break
        # if min_buddy is None:
        #     return min_edge, min_adj
        # else:
        #     return min_buddy, min_buddy_adj
        return min_edge, min_adj

    # returns the edges adjacent to a piece (in the existing group put together so far) if it were to be added
    def getAdjacentEdges(self, piece1, edge1, piece2, edge2):
        edges = []
        
        # get the info for the pieces
        piece1_loc, piece1_edge_up = self.info_dict[piece1]
        piece2_loc, piece2_edge_up = getPieceInfo(piece1, edge1, edge2, self.info_dict)

        # look in each direction
        directions = [(0,-1), (1,0), (0,1), (-1,0)]
        for i, d in enumerate(directions):
            # find the location of this piece, check if anything there
            piece3_loc = (piece2_loc[0] + d[0], piece2_loc[1] + d[1])
            if not self.position_dict.get(piece3_loc):
                continue
            # get the edge up, piece location of this other piece
            piece3, piece3_edge_up = self.position_dict.get(piece3_loc)
            # find the edges that would connect between the pieces
            if i == 0: # up
                edge2_2 = piece2_edge_up
                edge3 = (piece3_edge_up - 2) % 4
            elif i == 1: # right
                edge2_2 = (piece2_edge_up + 1) % 4
                edge3 = (piece3_edge_up - 1) % 4
            elif i == 2: # down
                edge2_2 = (piece2_edge_up + 2) % 4
                edge3 = piece3_edge_up
            else: # left
                edge2_2 = (piece2_edge_up - 1) % 4
                edge3 = (piece3_edge_up + 1) % 4
            # add to list of adjacent edges
            edges += [(piece3, edge3, piece2, edge2_2)]
        return edges

    # checks if the piece will make any changes to the existing known sides of the puzzle
    def updateEdges(self, piece, piece_loc, piece_edge_up):
        if piece.type == 'side' or piece.type == 'corner':
            # finds flat sides of the piece
            for i, edge in enumerate(piece.edges):
                edge_diff = (i - piece_edge_up) % 4
                if edge.label == 'flat':
                    # finds which side is flat relative to rest of puzzle, updates known side locations
                    edge_diff = (i - piece_edge_up) % 4
                    if edge_diff == 0:
                        self.top_edge = piece_loc[1]
                    elif edge_diff == 1:
                        self.right_edge = piece_loc[0]
                    elif edge_diff == 2:
                        self.bottom_edge = piece_loc[1]
                    else:
                        self.left_edge = piece_loc[0]

    def updatePieceDims(self, piece, piece_loc):
        min_x, max_x, min_y, max_y = self.all_piece_dims
        new_dims = (min(piece_loc[0], min_x), max(piece_loc[0], max_x), min(piece_loc[1], min_y), max(piece_loc[1], max_y))

        self.all_piece_dims = new_dims

        if piece.type == 'middle':
            if self.middle_piece_dims is None:
                self.middle_piece_dims = [piece_loc[0], piece_loc[0], piece_loc[1], piece_loc[1]]
            else:
                middle_min_x, middle_max_x, middle_min_y, middle_max_y = self.middle_piece_dims
                new_dims = (min(piece_loc[0], middle_min_x), max(piece_loc[0], middle_max_x),
                            min(piece_loc[1], middle_min_y), max(piece_loc[1], middle_max_y))
                self.middle_piece_dims = new_dims
        

    # checks if a piece would break any rules of a consistent, correctly put together puzzle if it were added
    def isValid(self, piece1, edge1, piece2, edge2):
        # if there is already a piece there, not valid
        piece2_loc, piece2_edge_up = getPieceInfo(piece1, edge1, edge2, self.info_dict)
        if self.position_dict.get(piece2_loc):
            return False

        # find width, height with and without the piece
        min_x, max_x, min_y, max_y = self.all_piece_dims

        min_x_with = min(min_x, piece2_loc[0])
        min_y_with = min(min_y, piece2_loc[1])
        max_x_with = max(max_x, piece2_loc[0])
        max_y_with = max(max_y, piece2_loc[1])

        width_without = max_x - min_x + 1
        height_without = max_y - min_y + 1

        width = max_x_with - min_x_with + 1
        height = max_y_with - min_y_with + 1

        # filter out middle pieces from pieces already placed
        if not self.middle_piece_dims is None:
            min_x_middle, max_x_middle, min_y_middle, max_y_middle = self.middle_piece_dims

            middle_width = max_x_middle - min_x_middle + 1
            middle_height = max_y_middle - min_y_middle + 1

                # update edges of puzzle according to rules of middle piece dimensions
            if middle_width == max(self.puzzle_dims) - 2 or (middle_width == min(self.puzzle_dims) - 2 and middle_height > min(self.puzzle_dims) - 2):
                self.left_edge = min_x_middle - 1
                self.right_edge = max_x_middle + 1

            if middle_height == max(self.puzzle_dims) - 2 or  (middle_height == min(self.puzzle_dims) - 2 and middle_width > min(self.puzzle_dims) - 2):
                self.top_edge = min_y_middle - 1
                self.bottom_edge = max_y_middle + 1
                
            if piece2.type == 'middle':
                # update middle width if this piece extends current bounds
                if piece2_loc[0] < min_x_middle:
                    middle_width += 1
                elif piece2_loc[0] > max_x_middle:
                    middle_width += 1
                if piece2_loc[1] < min_y_middle:
                    middle_height += 1
                elif piece2_loc[1] > max_y_middle:
                    middle_height += 1

                # if the middle piece is on the side of the puzzle, bad
                if self.right_edge == piece2_loc[0]:
                    return False
                elif self.left_edge == piece2_loc[0]:
                    return False

                if self.top_edge == piece2_loc[1]:
                    return False
                elif self.bottom_edge == piece2_loc[1]:
                    return False

                # if the middle piece extends the bounds of possible middle pieces too far, bad :(
                if max(middle_width, middle_height) > max(self.puzzle_dims) - 2:
                    return False
                if min(middle_width, middle_height) > min(self.puzzle_dims) - 2:
                    return False

                # if the middle piece extends the bounds, i.e. one edge is not defined, must be 1 less
                if max(width, height) > max(width_without, height_without) and max(width, height) > max(self.puzzle_dims) - 1:
                    return False
                elif min(width, height) > min(width_without, height_without) and min(width, height) > min(self.puzzle_dims) - 1:
                    return False

                # if the width and height are too big, bad
                if max(width, height) > max(self.puzzle_dims):
                    return False
                if min(width, height) > min(self.puzzle_dims):
                    return False
        else:
            middle_width = middle_height = 0

        if piece2.type == 'side' or piece2.type == 'corner':
            # once again, if width, height too big, bad
            if max(width, height) > max(self.puzzle_dims):
                return False

            if min(width, height) > min(self.puzzle_dims):
                return False

            # check each edge and find the flat one(s)
            for i, edge in enumerate(piece2.edges):
                if edge.label == 'flat':
                    orientation = (i - piece2_edge_up) % 4
                    if orientation == 0: # up
                        # first, looking at the top of the puzzle. If there is one, side goes up, it must line up
                        if self.top_edge != None:
                            if piece2_loc[1] != self.top_edge:
                                return False
                        elif piece2_loc[1] >= min_y: # if there is not a top side yet, this can't be at the min found so far
                            return False
                        # if the bottom edge exists, 
                        if self.bottom_edge != None:
                            # the height will be from this piece to the bottom edge
                            new_height = self.bottom_edge - piece2_loc[1] + 1
                            # if the side edges both exist, this new height must fit the defined dimensions
                            if self.right_edge != None and self.left_edge != None:
                                if width == max(self.puzzle_dims):
                                    if new_height != min(self.puzzle_dims):
                                        return False
                                else:
                                    if new_height != max(self.puzzle_dims):
                                        return False
                            elif not new_height in self.puzzle_dims: # otherwise, the height should at least be in the puzzle dimensions
                                return False
                        # not for corner pieces
                        if piece2.type == 'side':
                            # side pieces pointing up can't be on the left / right side of the puzzle. Corner pieces go there.
                            if self.right_edge == piece2_loc[0]:
                                return False
                            elif self.right_edge == piece2_loc[0] + max(self.puzzle_dims) - 1:
                                return False
                            elif self.right_edge == piece2_loc[0] + min(self.puzzle_dims) - 1:
                                if height > min(self.puzzle_dims) or middle_height > min(self.puzzle_dims) - 2:
                                    return False

                            if self.left_edge == piece2_loc[0]:
                                return False
                            elif self.left_edge == piece2_loc[0] - (max(self.puzzle_dims) - 1):
                                return False
                            elif self.left_edge == piece2_loc[0] - (min(self.puzzle_dims) - 1):
                                if height > min(self.puzzle_dims) or middle_height > min(self.puzzle_dims) - 2:
                                    return False

                            # if neither side edge has been defined, but this piece extends the width in a bad way
                            if self.right_edge is None and self.left_edge is None:
                                if width > max(self.puzzle_dims) - 2:
                                    return False
                                if height > max(self.puzzle_dims) - 2 and width > min(self.puzzle_dims) - 2:
                                    return False

                    elif orientation == 1: # same for up but right
                        if self.right_edge != None:
                            if piece2_loc[0] != self.right_edge:
                                return False
                        elif piece2_loc[0] <= max_x:
                            return False
                        if self.left_edge != None:
                            new_width = piece2_loc[0] - self.left_edge + 1
                            if self.bottom_edge != None and self.top_edge != None:
                                if height == max(self.puzzle_dims):
                                    if new_width != min(self.puzzle_dims):
                                        return False
                                else:
                                    if new_width != max(self.puzzle_dims):
                                        return False
                            elif not new_width in self.puzzle_dims:
                                return False
                        if piece2.type == 'side':
                            if self.bottom_edge == piece2_loc[1]:
                                return False
                            elif self.bottom_edge == piece2_loc[1] + max(self.puzzle_dims) - 1:
                                return False
                            elif self.bottom_edge == piece2_loc[1] + min(self.puzzle_dims) - 1:
                                if width > min(self.puzzle_dims) or middle_width > min(self.puzzle_dims) - 2:
                                    return False

                            if self.top_edge == piece2_loc[1]:
                                return False
                            elif self.top_edge == piece2_loc[1] - (max(self.puzzle_dims) - 1):
                                return False
                            elif self.top_edge == piece2_loc[1] - (min(self.puzzle_dims) - 1):
                                if width > min(self.puzzle_dims) or middle_width > min(self.puzzle_dims) - 2:
                                    return False
                            
                            if self.top_edge is None and self.bottom_edge is None:
                                if height > max(self.puzzle_dims) - 2:
                                    return False
                                if width > max(self.puzzle_dims) - 2 and height > min(self.puzzle_dims) - 2:
                                    return False
                                    
                    elif orientation == 2: # same for up but down
                        if self.bottom_edge != None:
                            if piece2_loc[1] != self.bottom_edge:
                                return False
                        elif piece2_loc[1] <= max_y:
                            return False
                        if self.top_edge != None:
                            new_height = piece2_loc[1] - self.top_edge + 1
                            if self.right_edge != None and self.left_edge != None:
                                if width == max(self.puzzle_dims):
                                    if new_height != min(self.puzzle_dims):
                                        return False
                                else:
                                    if new_height != max(self.puzzle_dims):
                                        return False
                            elif not new_height in self.puzzle_dims:
                                return False     
                        if piece2.type == 'side':
                            if self.right_edge == piece2_loc[0]:
                                return False
                            elif self.right_edge == piece2_loc[0] + max(self.puzzle_dims) - 1:
                                return False
                            elif self.right_edge == piece2_loc[0] + min(self.puzzle_dims) - 1:
                                if height > min(self.puzzle_dims) or middle_height > min(self.puzzle_dims) - 2:
                                    return False

                            if self.left_edge == piece2_loc[0]:
                                return False
                            elif self.left_edge == piece2_loc[0] - (max(self.puzzle_dims) - 1):
                                return False
                            elif self.left_edge == piece2_loc[0] - (min(self.puzzle_dims) - 1):
                                if height > min(self.puzzle_dims) or middle_height > min(self.puzzle_dims) - 2:
                                    return False
                            
                            if self.right_edge is None and self.left_edge is None:
                                if width > max(self.puzzle_dims) - 2:
                                    return False
                                if height > max(self.puzzle_dims) - 2 and width > min(self.puzzle_dims) - 2:
                                    return False
                    else: # same as for up but left
                        if self.left_edge != None:
                            if piece2_loc[0] != self.left_edge:
                                return False
                        elif piece2_loc[0] >= min_x:
                            return False
                        if self.right_edge != None:
                            new_width = self.right_edge - piece2_loc[0] + 1
                            if self.bottom_edge != None and self.top_edge != None:
                                if height == max(self.puzzle_dims):
                                    if new_width != min(self.puzzle_dims):
                                        return False
                                else:
                                    if new_width != max(self.puzzle_dims):
                                        return False
                            elif not new_width in self.puzzle_dims:
                                return False
                        if piece2.type == 'side':
                            if self.bottom_edge == piece2_loc[1]:
                                return False
                            elif self.bottom_edge == piece2_loc[1] + max(self.puzzle_dims) - 1:
                                return False
                            elif self.bottom_edge == piece2_loc[1] + min(self.puzzle_dims) - 1:
                                if width > min(self.puzzle_dims) or middle_width > min(self.puzzle_dims) - 2:
                                    return False

                            if self.top_edge == piece2_loc[1]:
                                return False
                            elif self.top_edge == piece2_loc[1] - (max(self.puzzle_dims) - 1):
                                return False
                            elif self.top_edge == piece2_loc[1] - (min(self.puzzle_dims) - 1):
                                if width > min(self.puzzle_dims) or middle_width > min(self.puzzle_dims) - 2:
                                    return False
                            
                            if self.top_edge is None and self.bottom_edge is None:
                                if height > max(self.puzzle_dims) - 2:
                                    return False
                                if width > max(self.puzzle_dims) - 2 and height > min(self.puzzle_dims) - 2:
                                    return False
        return True # if none of the above conditions were a thing, return true
                    
    # returns an image containing the solution to the puzzle
    def getSolutionImage(self, with_details=False):
        # get the bounds of the solution
        keys = self.position_dict.keys()
        min_x = min(keys, key=lambda x:x[0])[0]
        max_x = max(keys, key=lambda x:x[0])[0]

        min_y = min(keys, key=lambda x:x[1])[1]
        max_y = max(keys, key=lambda x:x[1])[1]

        # get the subimages for each piece in the solution facing the appropriate direction
        image_dict = {}
        max_size = 0
        for key in keys:
            piece, edge_up = self.position_dict[key]
            piece_image = piece.getSubimage(edge_up, with_details=with_details, resize_factor=0.4)
            image_dict[key] = piece_image

            piece_image_size = max(piece_image.shape)
            if piece_image_size > max_size:
                max_size = piece_image_size
        
        # add them all to the solution image
        w = max_x - min_x
        h = max_y - min_y
        solution_image = np.zeros((max_size * (h + 1), max_size * (w + 1), 3), dtype=np.uint8)
        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                piece_image = image_dict.get((x, y))
                if piece_image is None:
                    continue
                x_coord = (x - min_x) * max_size
                y_coord = (y - min_y) * max_size

                ph, pw, _ = piece_image.shape
                solution_image[y_coord:y_coord + ph, x_coord:x_coord + pw] = piece_image

        return solution_image        
        
# returns the location and edge facing up of piece2 with edge edge2 connnecting to edge1 on piece1
def getPieceInfo(piece1, edge1, edge2, info_dict):
    # up, right, down, left
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    # get the info for the first piece
    piece1_loc, piece1_edge_up = info_dict[piece1]
    # get the direction that the first edge is going
    edge_diff = (edge1 - piece1_edge_up) % 4
    direction = directions[edge_diff]
    # get the location for the second piece using properties of the piece's edges
    piece2_loc = (piece1_loc[0] + direction[0], piece1_loc[1] + direction[1])
    # find the appropriate edge up on piece 2
    if edge_diff == 0: # going up
        piece2_edge_up = (edge2 + 2) % 4
    elif edge_diff == 1: # going right
        piece2_edge_up = (edge2 + 1) % 4
    elif edge_diff == 2: # going down
        piece2_edge_up = edge2
    else: # going left
        piece2_edge_up = (edge2 - 1) % 4

    return piece2_loc, piece2_edge_up

if __name__=='__main__':
    main()

# so much code lol