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

    def crossover(self, other_solution, randomize=False):
        new_solution = Solution(self.pieces, self.solution_dimensions, self.dist_dict)
        # want a kernel of available spots to go

        kernel = set()
        available_pieces = set([piece for piece in self.pieces.pieces if piece.type == 'middle'])

        # pick a piece to begin the process from
        init_piece = random.choice(tuple(available_pieces))
        available_pieces.remove(init_piece)

        new_solution.position_dict[(0,0)] = (init_piece, 0)

        for pos in [(1,0), (-1,0), (0,1), (0,-1)]:
            kernel.add(pos)

        while len(available_pieces) > 0 and len(kernel) > 0:
            if randomize:
                tries = 500
                position, piece, edge_up = None, None, None
                while tries > 0:
                    position = random.choice(tuple(kernel))
                    piece = random.choice(tuple(available_pieces))
                    edge_up = random.choice(range(4))

                    if isValid(new_solution, position, piece, edge_up):
                        break
                    tries -= 1
                if tries == 0:
                    print('tried out :(')
                    position, piece, edge_up = None, None, None
            else:
                position, piece, edge_up = getNextEdge(available_pieces, kernel, self, other_solution, new_solution)

            if not piece:
                break

            new_solution.position_dict[position] = (piece, edge_up)

            piece_score = 0
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)] # up right down left
            for i, direction in enumerate(directions):
                adj_position = (direction[0] + position[0], direction[1] + position[1])
                value = new_solution.position_dict.get(adj_position)
                if not value:
                    kernel.add(adj_position)
                    continue
                adj_piece, adj_edge_up = value
                edge1 = (edge_up + i) % 4
                edge2 = (adj_edge_up + (i-2)) % 4
                diff = piece.edges[edge1].compare(adj_piece.edges[edge2])
                piece_score += diff
                self.edge_dict[(piece, edge1)] = (adj_piece, edge2)

            new_solution.score += piece_score

            kernel.remove(position)
            available_pieces.remove(piece)

        return new_solution

    def randomize(self):
        side_pieces = [piece for piece in self.pieces.pieces if piece.type == 'side']
        corner_pieces = [piece for piece in self.pieces.pieces if piece.type == 'corner']
        middle_pieces = [piece for piece in self.pieces.pieces if piece.type == 'middle']


        random.shuffle(side_pieces)
        random.shuffle(corner_pieces)
        random.shuffle(middle_pieces)



        for x in range(self.solution_dimensions[0]):
            for y in range(self.solution_dimensions[1]):
                piece = None
                edge_up = 0
                if x == 0:
                    if y == 0:
                        piece = corner_pieces.pop()
                        for i, edge in enumerate(piece.edges):
                            if edge.label == 'flat' and piece.edges[i-1].label == 'flat':
                                edge_up = i
                    elif y == self.solution_dimensions[1] - 1:
                        piece = corner_pieces.pop()
                        for i, edge in enumerate(piece.edges):
                            if edge.label == 'flat' and piece.edges[i-1].label == 'flat':
                                edge_up = (i + 1) % 4
                    else:
                        piece = side_pieces.pop()
                        for i, edge in enumerate(piece.edges):
                            if edge.label == 'flat':
                                edge_up = (i + 1) % 4
                elif x == self.solution_dimensions[0] - 1:
                    if y == 0:
                        piece = corner_pieces.pop()
                        for i, edge in enumerate(piece.edges):
                            if edge.label == 'flat' and piece.edges[i-1].label == 'flat':
                                edge_up = (i - 1) % 4
                    elif y == self.solution_dimensions[1] - 1:
                        piece = corner_pieces.pop()
                        for i, edge in enumerate(piece.edges):
                            if edge.label == 'flat' and piece.edges[i-1].label == 'flat':
                                edge_up = (i + 2) % 4
                    else:
                        piece = side_pieces.pop()
                        for i, edge in enumerate(piece.edges):
                            if edge.label == 'flat':
                                edge_up = (i - 1) % 4
                elif y == 0:
                    piece = side_pieces.pop()
                    for i, edge in enumerate(piece.edges):
                        if edge.label == 'flat':
                            edge_up = i
                elif y == self.solution_dimensions[1] - 1:
                    piece = side_pieces.pop()
                    for i, edge in enumerate(piece.edges):
                        if edge.label == 'flat':
                            edge_up = (i + 2) % 4
                else:
                    piece = middle_pieces.pop()
                    edge_up = random.choice(range(4))

                self.position_dict[(x,y)] = piece, edge_up
                if y > 0:
                    piece_above, above_edge_up = self.position_dict[(x, y-1)]
                    self.score += piece.edges[edge_up].compare(piece_above.edges[(above_edge_up + 2) % 4])
                if x > 0:
                    piece_left, left_edge_up = self.position_dict[(x-1, y)]
                    self.score += piece.edges[(edge_up - 1) % 4].compare(piece_left.edges[(left_edge_up + 1) % 4])



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

def isValid(new_solution, position, piece, edge_up):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)] # up right down left
    for i, direction in enumerate(directions):
        adj_position = (direction[0] + position[0], direction[1] + position[1])
        value = new_solution.position_dict.get(adj_position)
        if not value:
            continue
        adj_piece, adj_edge_up = value
        
        edge1_num = (edge_up + i) % 4
        edge2_num = (adj_edge_up + (i-2)) % 4

        edge1 = piece.edges[edge1_num]
        edge2 = adj_piece.edges[edge2_num]

        if edge1.label == 'flat' or edge2.label == 'flat':
            return False

        if (piece.edges[(edge1_num - 1) % 4].label == 'flat') != (adj_piece.edges[(edge2_num + 1) % 4].label == 'flat'):
            return False

        if (piece.edges[(edge1_num + 1) % 4].label == 'flat') != (adj_piece.edges[(edge2_num - 1) % 4].label == 'flat'):
            return False

    corner_keys = [key for key in new_solution.position_dict.keys() if new_solution.position_dict[key][0].type == 'corner']
    side_keys = [key for key in new_solution.position_dict.keys() if new_solution.position_dict[key][0].type != 'middle']
    middle_keys = [key for key in new_solution.position_dict.keys() if new_solution.position_dict[key][0].type == 'middle']

    top_side_y = bottom_side_y = right_side_x = left_side_x = None
    top_sides = bottom_sides = right_sides = left_sides = []
    for key in side_keys:
        side_piece, side_edge_up = new_solution.position_dict[key]

        if side_piece.edges[side_edge_up].label == 'flat':
            top_side_y = key[1]
            if side_piece.type == 'side':
                top_sides.append(key)

        if side_piece.edges[(side_edge_up + 1) % 4].label == 'flat':
            right_side_x = key[0]
            if side_piece.type == 'side':
                right_sides.append(key)

        if side_piece.edges[(side_edge_up + 2) % 4].label == 'flat':
            bottom_side_y = key[1]
            if side_piece.type == 'side':
                bottom_sides.append(key)

        if side_piece.edges[(side_edge_up + 3) % 4].label == 'flat':
            left_side_x = key[0]
            if side_piece.type == 'side':
                left_sides.append(key)

    
    if piece.edges[edge_up].label == 'flat':
        if top_side_y:
            if position[1] != top_side_y:
                return False
            max_top = max(top_sides + [position], key=lambda x:x[0])[0]
            min_top = min(top_sides + [position], key=lambda x:x[0])[0]
            if piece.type == 'side' and max_top - min_top > new_solution.solution_dimensions[0] - 2:
                return False

        if bottom_side_y and bottom_side_y - position[1] + 1 != new_solution.solution_dimensions[1]:
            # print(f'top: {bottom_side_y - position[1]}')
            return False

    if piece.edges[(edge_up + 1) % 4].label == 'flat':
        if right_side_x:
            if position[0] != right_side_x:
                return False
            max_right = max(right_sides + [position], key=lambda x:x[1])[1]
            min_right = min(right_sides + [position], key=lambda x:x[1])[1]
            if piece.type == 'side' and max_right - min_right > new_solution.solution_dimensions[1] - 2:
                return False
        if left_side_x and position[0] - left_side_x + 1 != new_solution.solution_dimensions[0]:
            # print(f'right: {position[0] - left_side_x}')
            return False
    if piece.edges[(edge_up + 2) % 4].label == 'flat':
        if bottom_side_y:
            if position[1] != bottom_side_y:
                return False
            max_bottom = max(bottom_sides + [position], key=lambda x:x[0])[0]
            min_bottom = min(bottom_sides + [position], key=lambda x:x[0])[0]
            if piece.type == 'side' and max_bottom - min_bottom > new_solution.solution_dimensions[0] - 2:
                return False
        if top_side_y and position[1] - top_side_y + 1 != new_solution.solution_dimensions[1]:
            # print(f'bottom: {position[1] - top_side_y}')
            return False
    if piece.edges[(edge_up + 3) % 4].label == 'flat':
        if left_side_x:
            if position[0] != left_side_x:
                return False
            max_left = max(left_sides + [position], key=lambda x:x[1])[1]
            min_left = min(left_sides + [position], key=lambda x:x[1])[1]
            if piece.type == 'side' and max_left - min_left > new_solution.solution_dimensions[1] - 2:
                return False

        if right_side_x and right_side_x - position[0] + 1 != new_solution.solution_dimensions[0]:
            # print(f'left: {right_side_x - position[0]}')
            return False

    if piece.type == 'middle':
        if position[1] == top_side_y or position[1] == bottom_side_y or position[0] == right_side_x or position[0] == left_side_x:
            return False
        
        if len(middle_keys) > 0:
            min_x = min(middle_keys, key=lambda x:x[0])[0]
            min_y = min(middle_keys, key=lambda x:x[1])[1]
            max_x = max(middle_keys, key=lambda x:x[0])[0]
            max_y = max(middle_keys, key=lambda x:x[1])[1]

            width = max_x - min_x + 1
            height = max_y - min_y + 1

            if position[0] < min_x or position[0] > max_x:
                if width + 1 > new_solution.solution_dimensions[0] - 2:
                    return False
            if position[1] < min_y or position[1] > max_y:
                if height + 1 > new_solution.solution_dimensions[1] - 2:
                    return False

    if piece.type == 'side':
        side_edges_rel = [i for i in range(4) if piece.edges[(edge_up + i) % 4].label == 'flat']
        for key in new_solution.position_dict.keys():
            other_piece, other_edge_up = new_solution.position_dict[key]
            other_side_edges_rel = [i for i in range(4) if other_piece.edges[(other_edge_up + i) % 4].label == 'flat']
            if key[0] == position[0]:
                if (1 in side_edges_rel) != (1 in other_side_edges_rel):
                    # print('1')
                    return False
                if (3 in side_edges_rel) != (3 in other_side_edges_rel):
                    # print('3')
                    return False
            if key[1] == position[1]:
                if (0 in side_edges_rel) != (0 in other_side_edges_rel):
                    # print('0')
                    return False
                if (2 in side_edges_rel) != (2 in other_side_edges_rel):
                    # print('2')
                    return False



        # min_x_tuple = min(new_solution.position_dict.keys(), key=lambda x:x[0])
        # max_x_tuple = max(new_solution.position_dict.keys(), key=lambda x:x[0])

        # min_y_tuple = min(new_solution.position_dict.keys(), key=lambda x:x[1])
        # max_y_tuple = max(new_solution.position_dict.keys(), key=lambda x:x[1])
        
        # width = max_x_tuple[0] - min_x_tuple[0] + 1
        # height = max_y_tuple[1] - min_y_tuple[1] + 1

        # max_width, max_height = new_solution.solution_dimensions
        
        # if new_solution.position_dict[min_x_tuple][0].label == 'middle':
        #     max_width -= 1
        # if new_solution.position_dict[max_x_tuple][0].label == 'middle':
        #     max_width -= 1
        # if new_solution.position_dict[min_y_tuple][0].label == 'middle':
        #     max_height -= 1
        # if new_solution.position_dict[max_y_tuple][0].label == 'middle':
        #     max_height -= 1

        # print(width, height, max_width, max_height)

        # if width > max_width:
        #     return False
        # if height > max_height:
        #     return False

        # if adj_position[0] < min_x_tuple[0] or adj_position[0] > max_x_tuple[0]:
        #     if width + 1 > max_width:
        #         return False
        # if adj_position[1] < min_y_tuple[1] or adj_position[1] > max_y_tuple[1]:
        #     if height + 1 > max_height:
        #         return False

    return True

def getNextEdge(available_pieces, kernel, curr_solution, other_solution, new_solution):
    min_score = float('inf')
    min_position = None
    min_piece = None
    min_edge_up = None

    for position in kernel:
        for piece in available_pieces:
            for edge_up in range(4):
                if not isValid(new_solution, position, piece, edge_up):
                    continue
                piece_score = 0
                directions = [(0, -1), (1, 0), (0, 1), (-1, 0)] # up right down left
                for i, direction in enumerate(directions):
                    adj_position = (direction[0] + position[0], direction[1] + position[1])
                    value = new_solution.position_dict.get(adj_position)
                    if not value:
                        continue
                    adj_piece, adj_edge_up = value
                    
                    edge1 = (edge_up + i) % 4
                    edge2 = (adj_edge_up + (i-2)) % 4

                    # if curr_solution.edge_dict.get((piece, edge1)) == (adj_piece, edge2) and other_solution.edge_dict.get((piece, edge1)) == (adj_piece, edge2):
                    #     return position, piece, edge_up

                    diff = curr_solution.dist_dict[(piece, edge1, adj_piece, edge2)]
                    piece_score += diff

                if piece_score < min_score:
                    min_score = piece_score
                    min_piece = piece
                    min_edge_up = edge_up
                    min_position = position
        
    return min_position, min_piece, min_edge_up

if __name__=='__main__':
    collection = PieceCollection()
    collection.addPieces('puzzle3_02.jpg', 20)
    collection.addPieces('puzzle3_01.jpg', 20)
    collection.addPieces('puzzle3_03.jpg', 20)
    collection.addPieces('puzzle3_04.jpg', 20)
    collection.addPieces('puzzle3_05.jpg', 20)


    random_solution = Solution(collection, (10,10))
    random_solution.randomize()

    cv2.imwrite(f'random_solution.jpg', random_solution.getSolutionImage())

    random_solution2 = Solution(collection, (10,10))
    random_solution2.randomize()

    cv2.imwrite(f'random_solution2.jpg', random_solution2.getSolutionImage())

    crossover_hit = random_solution.crossover(random_solution2)

    cv2.imwrite(f'crossover.jpg', crossover_hit.getSolutionImage())
    




