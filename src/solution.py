import random
import cv2
import numpy as np
from pieceCollection import PieceCollection
from copy import deepcopy

class Solution:
    def __init__(self, pieces, dimensions, dist_dict, buddy_dict):
        self.pieces = pieces
        self.solution_dimensions = dimensions
        self.position_dict = {}
        self.edge_dict = {}
        self.score = 0
        self.dist_dict = dist_dict
        self.buddy_dict = buddy_dict
        self.edge_set = set()

    def crossover(self, other_solution):
        new_solution = Solution(self.pieces, self.solution_dimensions, self.dist_dict, self.buddy_dict)

        kernel = set()
        possible_edges = set()
        piece_dict = {}

        available_pieces = set([piece for piece in self.pieces.pieces if piece.type == 'middle'])

        init_piece = random.choice(tuple(available_pieces))

        piece_dict[init_piece] = (0, 0, 0)
        available_pieces.remove(init_piece)
        new_solution.position_dict[(0,0)] = init_piece, 0

        for piece in available_pieces:
            for edge1 in range(4):
                for edge2 in range(4):
                    possible_edges.add((init_piece, edge1, piece, edge2))

        shared_edges = self.edge_set.intersection(other_solution.edge_set)

        while(len(available_pieces) > 0 and len(possible_edges) > 0):
            # print(len(available_pieces), len(possible_edges))

            possible_shared_edges = shared_edges.intersection(possible_edges)
            if len(possible_shared_edges) > 0:
                piece1, edge1, piece2, edge2 = min(possible_shared_edges, key=lambda x:self.dist_dict[x])
            else:
                buddies = [edge for edge in possible_edges if self.buddy_dict.get((edge[0], edge[1])) == (edge[2], edge[3])]
                if len(buddies) > 0:
                    piece1, edge1, piece2, edge2 = min(buddies, key=lambda x:self.dist_dict[x])
                else:
                    pieces = sorted(possible_edges, key=lambda x:self.dist_dict[x])[:5]
                    piece1, edge1, piece2, edge2 = random.choices(pieces, weights=[0.5, 0.3, 0.2, 0.1, 0.1], k=1)[0]

            piece1_x, piece1_y, piece1_edge_up = piece_dict[piece1]

            diff_edge_up = (edge1 - piece1_edge_up) % 4

            if diff_edge_up == 0: # going up
                edge_up = (edge2 + 2) % 4
                pos_x = piece1_x
                pos_y = piece1_y - 1
            elif diff_edge_up == 1: # going right
                edge_up = (edge2 + 1) % 4
                pos_x = piece1_x + 1
                pos_y = piece1_y
            elif diff_edge_up == 2: # going down
                edge_up = edge2
                pos_x = piece1_x
                pos_y = piece1_y + 1
            else: # going left
                edge_up = (edge2 - 1) % 4
                pos_x = piece1_x - 1
                pos_y = piece1_y

            if not isValid(new_solution, (pos_x, pos_y)):
                possible_edges.remove((piece1, edge1, piece2, edge2))
                continue

            piece_dict[piece2] = (pos_x, pos_y, edge_up)

            edges_to_remove = {edge for edge in possible_edges if (edge[0], edge[1]) == (piece1, edge1) or edge[2] == piece2}
            possible_edges = possible_edges.difference(edges_to_remove)
            available_pieces.remove(piece2)

            new_solution.position_dict[(pos_x, pos_y)] = (piece2, edge_up)
            new_solution.score += piece1.edges[edge1].compare(piece2.edges[edge2])
            new_solution.edge_set.add((piece1, edge1, piece2, edge2))

            # for i, direction in enumerate(([0, -1], [1, 0], [0, 1], [-1, 0])):
            #     pos_adj = (pos_x + direction[0], pos_y + direction[1])
            #     value = new_solution.position_dict.get(pos_adj)
            #     if not value:
            #         continue
            #     piece_adj, edge_up_adj = value
            #     edge_piece2 = (edge_up + i) % 4
            #     edge_adj = (edge_up_adj + i - 2) % 4
            #     new_solution.score += piece2.edges[edge_piece2].compare(piece_adj.edges[edge_adj])



            edges_to_add = set()
            for old_edge in range(4):
                if old_edge == edge2:
                    continue
                for new_piece in available_pieces:
                    for new_edge in range(4):
                        diff_edge_up = (old_edge - edge_up) % 4
                        if diff_edge_up == 0:
                            position = (pos_x, pos_y - 1)
                        elif diff_edge_up == 1:
                            position = (pos_x + 1, pos_y)
                        elif diff_edge_up == 2:
                            position = (pos_x, pos_y + 1)
                        else:
                            position = (pos_x - 1, pos_y)
                        if isValid(new_solution, position):
                            edges_to_add.add((piece2, old_edge, new_piece, new_edge))
            possible_edges = possible_edges.union(edges_to_add)

        edges_to_add = set()
        for piece1, edge1, piece2, edge2 in new_solution.edge_set:
            edges_to_add.add((piece2, edge2, piece1, edge1))
        new_solution.edge_set = new_solution.edge_set.union(edges_to_add)
        
        return new_solution

    def randomize(self):
        side_pieces = [piece for piece in self.pieces.pieces if piece.type == 'side']
        corner_pieces = [piece for piece in self.pieces.pieces if piece.type == 'corner']
        middle_pieces = [piece for piece in self.pieces.pieces if piece.type == 'middle']


        random.shuffle(side_pieces)
        random.shuffle(corner_pieces)
        random.shuffle(middle_pieces)

        for x in range(self.solution_dimensions[0] - 2):
            for y in range(self.solution_dimensions[1] - 2):
                piece = None
                edge_up = 0
                piece = middle_pieces.pop()
                edge_up = random.choice(range(4))

                self.position_dict[(x,y)] = piece, edge_up
                if y > 0:
                    piece_above, above_edge_up = self.position_dict[(x, y-1)]
                    self.score += piece.edges[edge_up].compare(piece_above.edges[(above_edge_up + 2) % 4])
                    self.edge_set.add((piece, edge_up, piece_above, (above_edge_up + 2) % 4))
                if x > 0:
                    piece_left, left_edge_up = self.position_dict[(x-1, y)]
                    self.score += piece.edges[(edge_up - 1) % 4].compare(piece_left.edges[(left_edge_up + 1) % 4])
                    self.edge_set.add((piece, (edge_up - 1) % 4, piece_left, (left_edge_up + 1) % 4))

        edges_to_add = set()
        for piece1, edge1, piece2, edge2 in self.edge_set:
            edges_to_add.add((piece2, edge2, piece1, edge1))
        self.edge_set = self.edge_set.union(edges_to_add)


    def getSolutionImage(self):
        keys = self.position_dict.keys()
        min_x = min(keys, key=lambda x:x[0])[0]
        max_x = max(keys, key=lambda x:x[0])[0]

        min_y = min(keys, key=lambda x:x[1])[1]
        max_y = max(keys, key=lambda x:x[1])[1]

        image_dict = {}
        max_size = 0
        for key in keys:
            piece, edge_up = self.position_dict[key]
            piece_image = piece.getSubimage(edge_up)
            image_dict[key] = piece_image

            piece_image_size = max(piece_image.shape)
            if piece_image_size > max_size:
                max_size = piece_image_size
        
        w = max_x - min_x
        h = max_y - min_y
        solution_image = np.zeros((max_size * (h + 1), max_size * (w + 1), 3), dtype=self.pieces.images[0].dtype)
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

def isValid(new_solution, position):

    if new_solution.position_dict.get(position):
        return False

    keys = new_solution.position_dict.keys()
    min_x = min(keys, key=lambda x:x[0])[0]
    min_y = min(keys, key=lambda x:x[1])[1]
    max_x = max(keys, key=lambda x:x[0])[0]
    max_y = max(keys, key=lambda x:x[1])[1]

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    if position[0] < min_x or position[0] > max_x:
        if width + 1 > new_solution.solution_dimensions[0] - 2:
            return False
    if position[1] < min_y or position[1] > max_y:
        if height + 1 > new_solution.solution_dimensions[1] - 2:
            return False
    return True

