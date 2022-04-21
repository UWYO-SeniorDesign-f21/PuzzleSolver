from pieceCollection import PieceCollection
import random
import cv2
import numpy as np
from timeit import default_timer as timer
import math
import time
import copy

class PuzzleSolution:
    def __init__(self, pieces, dims, dist_dict, sorted_dists, buddy_edges, empty_edge_dist, cutoff):
        self.pieces = pieces # pieceCollection object
        self.puzzle_dims = dims
        self.dist_dict = dist_dict # distance btw all edges
        self.sorted_dists = sorted_dists
        self.buddy_edges = buddy_edges

        self.position_dict = {} # x y coords to piece
        self.left_edge = self.right_edge = self.bottom_edge = self.top_edge = None
        self.info_dict = {} # piece to x y coords, edge up

        self.all_edges = set()
        self.all_edges_dict = {}
        self.edges = set()
        # self.close_edges = set()
        self.start = None

        self.score = 0
        self.similarity_score = 0
        self.empty_edge_dist = empty_edge_dist
        self.cutoff = cutoff

        self.parent_edges = set()

        self.all_piece_dims = [0,0,0,0]
        self.middle_piece_dims = None
        self.side_piece_dims = [None, None]

        self.randomness_factor = 1
        self.edge_cutoff = max(10, len(self.pieces.pieces) // 30)
        self.edge_cutoff_sides = max(5, len([piece for piece in self.pieces.pieces if piece.label in ['side', 'corner']]) // 10)

    def getDist(self, edge):
        if edge[0] == edge[2]:
            return float('inf')
        res = self.dist_dict.get(edge) if self.dist_dict.get(edge) else self.dist_dict.get((edge[2], edge[3], edge[0], edge[1]))
        if res is None:
            return float('inf')
        return res

    def crossover(self, other, just_sides=False):
        common_edges = self.edges.intersection(other.edges)
        # print(len(self.edges), len(other.edges), len(common_edges))
        # close_edges = self.close_edges.union(other.close_edges)
        # common_edges.union(close_edges)
        # print(len(common_edges), len(close_edges), len(include_edges))
        # connected_components = getConnectedComponents(common_edges)
        new_solution = PuzzleSolution(self.pieces, self.puzzle_dims, self.dist_dict, self.sorted_dists, self.buddy_edges, self.empty_edge_dist, cutoff=self.cutoff)
        new_solution.parent_edges = common_edges
        include_edges = sorted(list(common_edges), key=lambda x:self.getDist(x))
        # if len(connected_components) > 0:
        #     largest_component = list(connected_components[0])
        #     print(len(largest_component))
        #     new_solution.solvePuzzle(start=random.choice(largest_component), include_edges=common_edges)
        # else:
        start = random.choice(self.pieces.pieces)
        new_solution.solvePuzzle(start=start, include_edges=include_edges, just_sides=just_sides)
        return new_solution


    def copy(self):
        new_solution = PuzzleSolution(self.pieces, self.puzzle_dims, self.dist_dict, self.sorted_dists, self.buddy_edges, self.empty_edge_dist, cutoff=self.cutoff)

        new_solution.edges = self.edges.copy()
        new_solution.all_edges = self.all_edges.copy()
        # new_solution.close_edges = self.close_edges.copy()
        new_solution.info_dict = self.info_dict.copy()
        new_solution.position_dict = self.position_dict.copy()
        new_solution.score = self.score

        return new_solution

    def checkAllMutations(self):
        for piece1 in self.pieces.pieces:
            min_swap = None
            min_swap_cost = 0
            for piece2 in self.pieces.pieces:
                for edge1 in range(4):
                    for edge2 in range(4):
                        dist = self.swapEdges((piece1, edge1, piece2, edge2), just_cost_calc=True)  
                        if dist < min_swap_cost:
                            min_swap = (piece1, edge1, piece2, edge2)
                            min_swap_cost = dist
            if not min_swap is None:
                print(min_swap_cost)
                self.swapEdges(min_swap)
                self.score += min_swap_cost

    def mutate(self, mutation_rate, max_mutation_score=0):
        # if self.mutated:
        #     return self
        # roll = random.uniform(0, 1)
        # max_mutations = len(self.edges) // 2
        # num_mutations = 0
        # while roll <= mutation_rate and num_mutations < max_mutations:
        #     selection = random.choices(list(self.all_edges), k=min(len(self.edges), 32))
        #     remove_edge = max(selection, key=lambda x:self.dist_dict[(x[0], x[1])][(x[2], x[3])])
        #     remove_edges = [edge for edge in self.edges if edge[0] == remove_edge[0]]
        #     for edge in remove_edges:
        #         if edge in self.edges:
        #             self.edges.remove(edge)
        #             self.edges.remove((edge[2], edge[3], edge[0], edge[1]))
        #     roll = random.uniform(0, 1)
        #     num_mutations += 1
        # self.mutated = True
        # return self

        roll = random.uniform(0, 1)
        # new_solution = PuzzleSolution(self.pieces, self.puzzle_dims, self.dist_dict, self.sorted_dists, self.buddy_edges, self.empty_edge_dist, cutoff=self.cutoff)

        # new_solution.edges = self.edges.copy()
        # new_solution.all_edges = self.all_edges.copy()
        # new_solution.close_edges = self.close_edges.copy()
        # new_solution.info_dict = self.info_dict.copy()
        # new_solution.position_dict = self.position_dict.copy()
        # new_solution.score = self.score
        if roll > mutation_rate:         
            return

        total_mutation_cost = 0
        num_mutations = 0

        while roll <= mutation_rate:
            swap_cost = float('inf')
            swap = None
            # avoids bad swaps. these are rare, so probably will only iterate once
            tries = 0
            while swap_cost == float('inf') and tries < 10:
                swap = self.getSwap()
                if swap is None:
                    tries += 1
                    continue
                # will either update self.edges or return cost inf
                swap_cost = self.swapEdges(swap, just_cost_calc = True)
                tries += 1
            if total_mutation_cost + swap_cost <= max_mutation_score and swap_cost < max_mutation_score / 2:
                # print(swap_cost)
                self.swapEdges(swap)
                # print(self.score, swap_cost)
                self.score += swap_cost

                piece1, edge1, piece2, edge2 = swap

                piece1_pos_old, piece1_edge_up_old = self.info_dict[piece1]
                piece2_pos_old, piece2_edge_up_old = self.info_dict[piece2]
                diff_piece2 = piece2_edge_up_old - edge2
                diff_piece1 = piece1_edge_up_old - edge1
                piece1_edge_up_new = (edge1 + diff_piece2) % 4
                piece2_edge_up_new = (edge2 + diff_piece1) % 4
                self.info_dict[piece1] = (piece2_pos_old, piece1_edge_up_new)
                self.info_dict[piece2] = (piece1_pos_old, piece2_edge_up_new)
                self.position_dict[piece2_pos_old] = (piece1, piece1_edge_up_new)
                self.position_dict[piece1_pos_old] = (piece2, piece2_edge_up_new)

                total_mutation_cost += swap_cost

                num_mutations += 1

            roll = random.uniform(0, 1)
        # if total_mutation_cost < 0:
            # print(total_mutation_cost)
            # show_edges = self.all_edges - new_solution.all_edges
            # img1 = self.getSolutionImage(draw_edges=list(show_edges))
            # img2 = new_solution.getSolutionImage(draw_edges=list(show_edges))

            # cv2.imwrite(f'pre_mutation_score{self.score}.jpg', img1)
            # cv2.imwrite(f'post_mutationn_score{new_solution.score}.jpg', img2)
            # cv2.imshow('ijmg2', img2)
            # cv2.waitKey(0)
            # return self

    def getSwap(self):
        worst_edges = set()
        side_mutation = random.uniform(0,1)
        if side_mutation < 0.1:
            allowed_edges = [edge for edge in self.all_edges if edge[0].type in ['side', 'corner'] and edge[2].type in ['side', 'corner']]
        else:
            allowed_edges = [edge for edge in self.all_edges]
        if len(allowed_edges) == 0:
            return None
        tries = 0
        while len(worst_edges) < max(1, len(allowed_edges) // 100) and tries < len(allowed_edges) / 32:
            selection = random.sample(allowed_edges, k=max(1, len(allowed_edges) // 100))
            worst_edge = max(selection, key=lambda x:self.getDist(x))
            if not worst_edge in worst_edges:
                worst_edges.add(worst_edge)
            tries += 1
        dists = []
        # all_edges = sorted(list(self.all_edges), key=lambda x:-self.dist_dict[(x[0], x[1])][(x[2], x[3])])
        # worst_edges = all_edges[:16]
        for piece1, piece1_edge, piece3, piece3_edge in worst_edges:
            for piece2, piece2_edge, piece4, piece4_edge in worst_edges:
                if piece1 == piece2 and piece1_edge == piece2_edge:
                    continue
                dist = self.swapEdges((piece1, piece1_edge, piece2, piece2_edge), just_cost_calc=True)
                dists.append((piece1, piece1_edge, piece2, piece2_edge, dist))
        
        if len(dists) > 0:
            p1, e1, p2, e2, _ = min(dists, key=lambda x:x[4])
            swap = (p1, e1, p2, e2)
        else:
            return None

        # random_edges = random.sample(self.edges, k=min(len(self.edges), 64))
        # piece1, edge1, piece3, edge3 = max(random_edges, key=lambda x:self.dist_dict[(x[0], x[1])][x[2], x[3]])

        # sorted_dists = self.sorted_dists[(piece3, edge3)]
        # piece2_choices = random.sample(sorted_dists, k=min(len(sorted_dists), 64))
        # piece2, edge2 = min(piece2_choices, key=lambda x:x[1])[0]
        # swap = (piece1, edge1, piece2, edge2)
        return swap

    # swaps the edges and returns the net cost of the swap
    def swapEdges(self, swap, just_cost_calc=False):
        swap_cost = 0
        piece1_swap, edge1_swap, piece2_swap, edge2_swap = swap

        # if piece1_swap == piece2_swap:
        #     return float('inf')
        
        add_edges = set()
        remove_edges = set()
        for i in range(4):
            piece1_edge = (edge1_swap + i) % 4
            piece2_edge = (edge2_swap + i) % 4

            if (piece1_swap.edges[piece1_edge].label == "flat") ^ (piece2_swap.edges[piece2_edge].label == "flat"):
                return float('inf')


            entry = self.all_edges_dict.get((piece1_swap, piece1_edge))
            if not entry is None:
                piece3, piece3_edge = entry
            else:
                continue
            
            entry = self.all_edges_dict.get((piece2_swap, piece2_edge))
            if not entry is None:
                piece4, piece4_edge = entry
            else:
                continue

            # swap those edges >:(
            if not (piece1_swap, piece1_edge, piece3, piece3_edge) in remove_edges:
                remove_edges.add((piece1_swap, piece1_edge, piece3, piece3_edge))
                remove_edges.add((piece3, piece3_edge, piece1_swap, piece1_edge))
                swap_cost -= self.getDist((piece1_swap, piece1_edge, piece3, piece3_edge))

            if not (piece2_swap, piece2_edge, piece4, piece4_edge) in remove_edges:
                remove_edges.add((piece2_swap, piece2_edge, piece4, piece4_edge))
                remove_edges.add((piece4, piece4_edge, piece2_swap, piece2_edge))
                swap_cost -= self.getDist((piece2_swap, piece2_edge, piece4, piece4_edge))

            if not (piece1_swap, piece1_edge, piece4, piece4_edge) in add_edges:
                add_edges.add((piece1_swap, piece1_edge, piece4, piece4_edge))
                add_edges.add((piece4, piece4_edge, piece1_swap, piece1_edge))
                swap_cost += self.getDist((piece1_swap, piece1_edge, piece4, piece4_edge))

            if not (piece2_swap, piece2_edge, piece3, piece3_edge) in add_edges:
                add_edges.add((piece2_swap, piece2_edge, piece3, piece3_edge))
                add_edges.add((piece3, piece3_edge, piece2_swap, piece2_edge))
                swap_cost += self.getDist((piece2_swap, piece2_edge, piece3, piece3_edge))
        
        if swap_cost == 0 or swap_cost == float('inf'):
            return float('inf')

        if just_cost_calc:
            return swap_cost
        for edge in remove_edges:
            self.all_edges.remove(edge)
            self.all_edges_dict[(edge[0], edge[1])] = None
            if edge in self.edges:
                self.edges.remove(edge)
        for edge in add_edges:
            self.all_edges.add(edge)
            self.all_edges_dict[(edge[0], edge[1])] = (edge[2], edge[3])
            dist = self.getDist(edge)
            if ((edge[2], edge[3]), dist) in self.sorted_dists[(edge[0], edge[1])][:self.edge_cutoff]:
                self.edges.add(edge)
        return swap_cost

    def solvePuzzle(self, random_start=False, show_solve=False, start=None, include_edges=[], do_best_buddies=True, just_sides=False):
        if not start is None:
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
        kernel_num_edges = [ListDict(), ListDict(), ListDict(), ListDict()]
        best_buddy_num_edges = [ListDict(), ListDict(), ListDict(), ListDict()]
        include_num_edges =  [ListDict(), ListDict(), ListDict(), ListDict()]
        best_buddy_starts = set((edge[0], edge[1]) for edge in self.buddy_edges)
        best_buddy_ends = {(edge[0], edge[1]) : (edge[0], edge[1], edge[2], edge[3]) for edge in self.buddy_edges}
        include_starts = set((edge[0], edge[1]) for edge in include_edges)
        include_ends = {(edge[0], edge[1]) : (edge[0], edge[1], edge[2], edge[3]) for edge in include_edges}

        for edge in range(4):
            if piece1.edges[edge].label != 'flat':
                if not just_sides or (piece1.edges[(edge - 1) % 4].label == 'flat' or piece1.edges[(edge + 1) % 4].label == 'flat'):
                    kernel.add((piece1, edge))
                    kernel_num_edges[0].add((piece1, edge))
                    if (piece1, edge) in include_starts:
                        include_num_edges[0].add(include_ends[(piece1, edge)])
                    elif (piece1, edge) in best_buddy_starts:
                        best_buddy_num_edges[0].add(best_buddy_ends[(piece1, edge)])

        if show_solve:
            image = self.getSolutionImage(with_details=False)
            h, w, _  = image.shape
            cv2.imshow(f'solution', cv2.resize(image, (int(500 * (w / h)), 500), interpolation=cv2.INTER_AREA))
            cv2.waitKey(1)
            cv2.imwrite(f'solution{len(remaining_pieces)}.jpg', image)

        # while pieces are able to be added
        while len(remaining_pieces) > 0 and len(kernel) > 0:
            # print(len(remaining_pieces), len(kernel))
            min_edge = None
            min_adj = None
            min_dist = float('inf')
            if len(include_edges) > 0:
                min_edge, min_adj, min_dist = self.getNextEdgeInclude(include_edges, include_num_edges, remaining_pieces)
            if min_dist == float('inf') and do_best_buddies:
                min_edge, min_adj, min_dist = self.getNextEdgeInclude(self.buddy_edges, best_buddy_num_edges, remaining_pieces)
            if min_dist == float('inf'):
                min_edge, min_adj, min_dist = self.getNextEdge(kernel_num_edges, remaining_pieces)
            if min_dist == float('inf'): # if there are no possible piece locations
                break
            # get the closest edge
            piece1, edge1, piece2, edge2 = min_edge
            remaining_pieces.remove(piece2) # update available pieces, as piece2 is added to group

            self.score += min_dist
            
            # update the score, kernel
            for edge in min_adj:
                dist = self.getDist(edge)
                if edge[0].type in ['side', 'corner'] and edge[2].type in ['side', 'corner']:
                    edge_cutoff = self.edge_cutoff_sides
                else:
                    edge_cutoff = self.edge_cutoff
                if ((edge[2], edge[3]), dist) in self.sorted_dists[(edge[0], edge[1])][:edge_cutoff]:
                    self.edges.add(edge)
                    self.edges.add((edge[2], edge[3], edge[0], edge[1]))
                # dist = self.dist_dict[edge[0], edge[1]][edge[2], edge[3]]
                # self.score += dist
                # if ((edge[2], edge[3]), dist) in self.sorted_dists[(edge[0], edge[1])][:self.edge_cutoff]:
                    # self.edges.add(edge)
                    # self.edges.add((edge[2], edge[3], edge[0], edge[1]))
                    # if dist < self.cutoff:
                    #     self.close_edges.add(edge)
                    #     self.close_edges.add((edge[2], edge[3], edge[0], edge[1]))
                self.all_edges.add(edge)
                self.all_edges.add((edge[2], edge[3], edge[0], edge[1]))
                if (edge[0], edge[1]) in kernel or (edge[2], edge[3]) in kernel:
                    kernel.remove((edge[0], edge[1]))
                    for num_adj in range(4):
                        if (edge[0], edge[1]) in kernel_num_edges[num_adj]:
                            kernel_num_edges[num_adj].remove((edge[0], edge[1]))
                        if include_ends.get((edge[0], edge[1])) in include_num_edges[num_adj]:
                            include_num_edges[num_adj].remove(include_ends[(edge[0], edge[1])])
                        if best_buddy_ends.get((edge[0], edge[1])) in best_buddy_num_edges[num_adj]:
                            best_buddy_num_edges[num_adj].remove(best_buddy_ends[(edge[0], edge[1])])
                else:
                    print('zoinks')
                    
            # get the location and edge up of piece 2, update info dict, position dict
            piece1_loc, piece1_edge_up = self.info_dict[piece1]
            piece2_loc, piece2_edge_up = getPieceInfo(piece1, edge1, edge2, piece1_loc, piece1_edge_up)

            self.info_dict[piece2] = (piece2_loc, piece2_edge_up)
            self.position_dict[piece2_loc] = (piece2, piece2_edge_up)

            # add edges that are facing an empty spot to kernel
            used_edges = [edge[3] for edge in min_adj]
            for edge in range(4):
                if edge not in used_edges:
                    new_piece_loc, new_piece_edge_up = getPieceInfo(piece2, edge, 0, piece2_loc, piece2_edge_up)
                    if self.position_dict.get(new_piece_loc):
                        continue
                    elif piece2.edges[edge].label != 'flat':
                        if not just_sides or (piece2.edges[(edge - 1) % 4].label == 'flat' or piece2.edges[(edge + 1) % 4].label == 'flat'):
                            kernel.add((piece2, edge))
                            num_adj = len(self.getAdjacentEdges(piece2, edge, None, 0, piece2_loc, piece2_edge_up)[0])
                            
                            for prev_num_adj in range(4):
                                if (piece2, edge) in kernel_num_edges[prev_num_adj]:
                                    kernel_num_edges[prev_num_adj].remove((piece2, edge))
                                if include_ends.get((piece2, edge)) in include_num_edges[prev_num_adj]:
                                    include_num_edges[num_adj].remove(include_ends[(piece2, edge)])
                                if best_buddy_ends.get((piece2, edge)) in best_buddy_num_edges[prev_num_adj]:
                                    best_buddy_num_edges[num_adj].remove(best_buddy_ends[(piece2, edge)])
                            kernel_num_edges[num_adj - 1].add((piece2, edge))
                            if (piece2, edge) in include_starts:
                                include_num_edges[num_adj - 1].add(include_ends[(piece2, edge)])
                            elif (piece2, edge) in best_buddy_starts:
                                best_buddy_num_edges[num_adj - 1].add(best_buddy_ends[(piece2, edge)])

            # get the locations of the left, right, top, and bottom edges of the puzzle if applicable
            self.updateEdges(piece2, piece2_loc, piece2_edge_up)
            self.updatePieceDims(piece2, piece2_loc, piece2_edge_up)
            if show_solve:
                image = self.getSolutionImage(with_details=False)
                h, w, _  = image.shape
                cv2.imshow(f'solution', cv2.resize(image, (int(500 * (w / h)), 500), interpolation=cv2.INTER_AREA))
                cv2.waitKey(1)
                cv2.imwrite(f'solution{len(remaining_pieces)}.jpg', image)
        # for any empty spots, penalty
        # if len(remaining_pieces) != 0:
        #     solution_image = self.getSolutionImage()
        #     cv2.imshow('sol', cv2.resize(solution_image, (500,500), interpolation=cv2.INTER_AREA))
        #     cv2.waitKey(0)

        self.score += 4*len(remaining_pieces)*self.empty_edge_dist
        self.all_edges_dict = {(edge[0], edge[1]):(edge[2], edge[3]) for edge in self.all_edges}

    def getNextEdge(self, kernel_num_edges, remaining_pieces):
        min_edge = None
        min_adj = None
        min_dist = float('inf')
        # loop over possible edges to extend and possible edges to connect to them
        for num_adj in [4,3,2,1]:
            kernel_num_adj = kernel_num_edges[num_adj - 1]

            if len(kernel_num_adj) == 0:
                continue

            for piece1, edge1 in kernel_num_adj:

                piece1_loc, piece1_edge_up = self.info_dict[piece1]

                for edge, dist in self.sorted_dists[(piece1, edge1)]:
                    if dist >= min_dist:
                        break

                    piece2, edge2 = edge
                    if piece2 not in remaining_pieces:
                        continue

                    adj_edges, piece2_pos = self.getAdjacentEdges(piece1, edge1, piece2, edge2, piece1_loc, piece1_edge_up)
                    dist = 0
                    valid = True

                    for adj_edge in adj_edges:
                        dist += self.getDist(adj_edge)

                    if dist >= min_dist:
                        continue

                    for adj_edge in adj_edges:
                        if not self.isValid(adj_edge[0], adj_edge[1], adj_edge[2], adj_edge[3], piece2_pos[0], piece2_pos[1]):
                            valid = False
                            break

                    if not valid:
                        continue   

                    min_dist = dist
                    min_edge = (piece1, edge1, piece2, edge2)
                    min_adj = adj_edges

                    if not min_edge is None and num_adj == 1:
                        break

                if min_dist == float('inf'):
                    kernel_num_edges[num_adj-1].remove((piece1, edge1))
                else:
                    break

            if min_dist != float('inf'):
                break

        return min_edge, min_adj, min_dist

    def getNextEdgeInclude(self, edges_to_include, include_num_edges, remaining_pieces):
        min_edge = None
        min_adj = None
        min_dist = float('inf')
        # min_buddy = None
        # min_buddy_adj = None
        # min_buddy_dist = float('inf')
        # valid_edges = [edge for edge in edges_to_include if edge[2] in remaining_pieces and not kernel_num_edges.get((edge[0], edge[1])) is None]
        # for num_adj in [4, 3, 2, 1]:
        # edges_num_adj = [edge for edge in valid_edges if kernel_num_edges.get((edge[0], edge[1])) == num_adj]
        # loop over possible edges to extend and possible edges to connect to them
        
        for num_adj in [4,3,2,1]:
            edge_set = include_num_edges[num_adj - 1]
            if len(edge_set) == 0:
                continue
            for edge in edge_set:
                piece1, edge1, piece2, edge2 = edge
                if piece2 not in remaining_pieces:
                    include_num_edges[num_adj - 1].remove((piece1, edge1, piece2, edge2))
                    continue
                
                piece1_loc, piece1_edge_up = self.info_dict[piece1]

                adj_edges, piece2_pos = self.getAdjacentEdges(piece1, edge1, piece2, edge2, piece1_loc, piece1_edge_up)

                dist = 0
                # check if each edge is valid, find the distance
                for adj_edge in adj_edges:
                    adj_dist = self.getDist(adj_edge)
                    dist += adj_dist
                
                if dist == float('inf'):
                    include_num_edges[num_adj - 1].remove((piece1, edge1, piece2, edge2))
                    continue

                valid = True
                for adj_edge in adj_edges:
                    if not self.isValid(adj_edge[0], adj_edge[1], adj_edge[2], adj_edge[3], piece2_pos[0], piece2_pos[1]):
                        valid = False
                        break

                if not valid:
                    include_num_edges[num_adj - 1].remove((piece1, edge1, piece2, edge2))
                    continue

                min_dist = dist
                min_edge = (piece1, edge1, piece2, edge2)
                min_adj = adj_edges
                break

            if not min_dist == float('inf'):
                break
                
        return min_edge, min_adj, min_dist

    # returns the edges adjacent to a piece (in the existing group put together so far) if it were to be added
    def getAdjacentEdges(self, piece1, edge1, piece2, edge2, piece1_loc, piece1_edge_up):
        edges = []
        
        # get the info for the pieces
        piece2_loc, piece2_edge_up = getPieceInfo(piece1, edge1, edge2, piece1_loc, piece1_edge_up)

        # look in each direction
        directions = [(0,-1), (1,0), (0,1), (-1,0)]
        for i, d in enumerate(directions):
            # find the location of this piece, check if anything there
            piece3_loc = (piece2_loc[0] + d[0], piece2_loc[1] + d[1])
            entry = self.position_dict.get(piece3_loc)
            if entry is None:
                continue
            # get the edge up, piece location of this other piece
            piece3, piece3_edge_up = entry
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

        return edges, (piece2_loc, piece2_edge_up)

    # checks if the piece will make any changes to the existing known sides of the puzzle
    def updateEdges(self, piece, piece_loc, piece_edge_up):
        if piece.type == 'side' or piece.type == 'corner':
            # finds flat sides of the piece
            for i, edge in enumerate(piece.edges):
                edge_diff = (i - piece_edge_up) % 4
                if edge.label == 'flat':
                    # finds which side is flat relative to rest of puzzle, updates known side locations
                    # edge_diff = (i - piece_edge_up) % 4
                    if edge_diff == 0:
                        self.top_edge = piece_loc[1]
                    elif edge_diff == 1:
                        self.right_edge = piece_loc[0]
                    elif edge_diff == 2:
                        self.bottom_edge = piece_loc[1]
                    else:
                        self.left_edge = piece_loc[0]

            for i, edge in enumerate(piece.edges):
                edge_diff = (i - piece_edge_up) % 4
                if edge.label == 'flat' or piece.type == 'middle':
                    if edge_diff == 0 or edge_diff == 2:
                        if not self.right_edge is None and self.left_edge is None:
                            if self.right_edge - piece_loc[0] >= min(self.puzzle_dims):
                                self.left_edge = self.right_edge - (max(self.puzzle_dims) - 1)
                                if self.top_edge is None and not self.bottom_edge is None:
                                    self.top_edge = self.bottom_edge - (min(self.puzzle_dims) - 1)
                                if self.bottom_edge is None and not self.top_edge is None:
                                    self.bottom_edge = self.top_edge + (min(self.puzzle_dims) - 1)
                                
                        elif not self.left_edge is None and self.right_edge is None:
                            if piece_loc[0] - self.left_edge >= min(self.puzzle_dims):
                                self.right_edge = self.left_edge + (max(self.puzzle_dims) - 1)
                                if self.top_edge is None and not self.bottom_edge is None:
                                    self.top_edge = self.bottom_edge - (min(self.puzzle_dims) - 1)
                                if self.bottom_edge is None and not self.top_edge is None:
                                    self.bottom_edge = self.top_edge + (min(self.puzzle_dims) - 1)
                                
                    elif edge_diff == 1 or edge_diff == 3:
                        if not self.bottom_edge is None and self.top_edge is None:
                            if self.bottom_edge - piece_loc[1] >= min(self.puzzle_dims):
                                self.top_edge = self.bottom_edge - (max(self.puzzle_dims) - 1)
                                if self.left_edge is None and not self.right_edge is None:
                                    self.left_edge = self.right_edge - (min(self.puzzle_dims) - 1)
                                if self.right_edge is None and not self.left_edge is None:
                                    self.right_edge = self.left_edge + (min(self.puzzle_dims) - 1)
                                
                        elif not self.top_edge is None and self.bottom_edge is None:
                            if piece_loc[1] - self.top_edge >= min(self.puzzle_dims):
                                self.bottom_edge = self.top_edge + (max(self.puzzle_dims) - 1)
                                if self.left_edge is None and not self.right_edge is None:
                                    self.left_edge = self.right_edge - (min(self.puzzle_dims) - 1)
                                if self.right_edge is None and not self.left_edge is None:
                                    self.right_edge = self.left_edge + (min(self.puzzle_dims) - 1)
                                
        
        elif not self.middle_piece_dims is None:
            min_x_middle, max_x_middle, min_y_middle, max_y_middle = self.middle_piece_dims

            middle_width = max_x_middle - min_x_middle + 1
            middle_height = max_y_middle - min_y_middle + 1

                # update edges of puzzle according to rules of middle piece dimensions
            if middle_width == max(self.puzzle_dims) - 2 or (middle_width == min(self.puzzle_dims) - 2 and middle_height > min(self.puzzle_dims) - 2):
                self.left_edge = min_x_middle - 1
                self.right_edge = max_x_middle + 1

            if middle_height == max(self.puzzle_dims) - 2 or (middle_height == min(self.puzzle_dims) - 2 and middle_width > min(self.puzzle_dims) - 2):
                self.top_edge = min_y_middle - 1
                self.bottom_edge = max_y_middle + 1

    def updatePieceDims(self, piece, piece_loc, piece_edge_up):
        min_x, max_x, min_y, max_y = self.all_piece_dims
        new_dims = (min(piece_loc[0], min_x), max(piece_loc[0], max_x), min(piece_loc[1], min_y), max(piece_loc[1], max_y))

        self.all_piece_dims = new_dims

        if piece.type == 'middle':
            if self.middle_piece_dims is None:
                self.middle_piece_dims = (piece_loc[0], piece_loc[0], piece_loc[1], piece_loc[1])
            else:
                middle_min_x, middle_max_x, middle_min_y, middle_max_y = self.middle_piece_dims
                new_dims = (min(piece_loc[0], middle_min_x), max(piece_loc[0], middle_max_x),
                            min(piece_loc[1], middle_min_y), max(piece_loc[1], middle_max_y))
                self.middle_piece_dims = new_dims
        

        if not self.side_piece_dims[0] is None:
            if not self.left_edge is None:
                self.side_piece_dims[0] = (self.left_edge + 1, self.side_piece_dims[0][1])
            if not self.right_edge is None:
                self.side_piece_dims[0] = (self.side_piece_dims[0][0], self.right_edge - 1)
        if not self.side_piece_dims[1] is None:
            if not self.top_edge is None:
                self.side_piece_dims[1] = (self.top_edge + 1, self.side_piece_dims[1][1])
            if not self.bottom_edge is None:
                self.side_piece_dims[1] = (self.side_piece_dims[1][0], self.bottom_edge - 1)

        if piece.type == 'side':
            flat_edge = min([i for i in range(4) if piece.edges[i].label == 'flat'])
            edge_diff = (flat_edge - piece_edge_up) % 4
            if self.side_piece_dims[edge_diff % 2] is None:
                self.side_piece_dims[edge_diff % 2] = (piece_loc[edge_diff % 2], piece_loc[edge_diff % 2])
            else:
                side_min, side_max = self.side_piece_dims[edge_diff % 2]
                new_dims = (min(piece_loc[edge_diff % 2], side_min), max(piece_loc[edge_diff % 2], side_max))
                self.side_piece_dims[edge_diff % 2] = new_dims

    # checks if a piece would break any rules of a consistent, correctly put together puzzle if it were added
    def isValid(self, piece1, edge1, piece2, edge2, piece2_loc, piece2_edge_up, debug=False):

        if self.position_dict.get(piece2_loc):
            if debug:
                print(self.all_piece_dims, '1')
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

        if piece2.type == 'side':

            if not self.side_piece_dims[0] is None and not self.side_piece_dims[1] is None:
                hor_sides_min, hor_sides_max = self.side_piece_dims[0]
                vert_sides_min, vert_sides_max = self.side_piece_dims[1]


                if piece2.edges[piece2_edge_up].label == 'flat' or piece2.edges[(piece2_edge_up + 2) % 4].label == 'flat':
                    hor_sides_min = min(hor_sides_min, piece2_loc[0])
                    hor_sides_max = max(hor_sides_max, piece2_loc[0])
                else:                         
                    vert_sides_min = min(vert_sides_min, piece2_loc[1])
                    vert_sides_max = max(vert_sides_max, piece2_loc[1])

                hor_width = hor_sides_max - hor_sides_min + 1
                vert_width = vert_sides_max - vert_sides_min + 1

                if hor_width > max(self.puzzle_dims) - 2 or vert_width > max(self.puzzle_dims) - 2:
                    if debug:
                        print(self.side_piece_dims, '1.2')
                    return False

                if hor_width > min(self.puzzle_dims) - 2 and vert_width > min(self.puzzle_dims) - 2:
                    if debug:
                        print(self.side_piece_dims, '1.3')
                    return False

        # filter out middle pieces from pieces already placed
        if not self.middle_piece_dims is None:

            min_x_middle, max_x_middle, min_y_middle, max_y_middle = self.middle_piece_dims

            middle_width = max_x_middle - min_x_middle + 1
            middle_height = max_y_middle - min_y_middle + 1

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
                    if debug:
                        print(self.all_piece_dims, '2')
                    return False
                elif self.left_edge == piece2_loc[0]:
                    if debug:
                        print(self.all_piece_dims, '3')
                    return False
                if self.top_edge == piece2_loc[1]:
                    if debug:
                        print(self.all_piece_dims, '4')
                    return False
                elif self.bottom_edge == piece2_loc[1]:
                    if debug:
                        print(self.all_piece_dims, '5')
                    return False

                # if the middle piece extends the bounds of possible middle pieces too far, bad :(
                if max(middle_width, middle_height) > max(self.puzzle_dims) - 2:
                    if debug:
                        print(self.all_piece_dims, '6')
                    return False
                if min(middle_width, middle_height) > min(self.puzzle_dims) - 2:
                    if debug:
                        print(self.all_piece_dims, '7')
                    return False

                # if the middle piece extends the bounds, i.e. one edge is not defined, must be 1 less
                if max(width, height) > max(width_without, height_without) and max(width, height) > max(self.puzzle_dims) - 1:
                    if debug:
                        print(self.all_piece_dims, '8')
                    return False
                elif min(width, height) > min(width_without, height_without) and min(width, height) > min(self.puzzle_dims) - 1:
                    if debug:
                        print(self.all_piece_dims, '9')
                    return False

                # if the width and height are too big, bad
                if max(width, height) > max(self.puzzle_dims):
                    if debug:
                        print(self.all_piece_dims, '10')
                    return False
                if min(width, height) > min(self.puzzle_dims):
                    if debug:
                        print(self.all_piece_dims, '11')
                    return False
        else:
            middle_width = middle_height = 0

        if piece2.type == 'side' or piece2.type == 'corner':
            # once again, if width, height too big, bad
            if max(width, height) > max(self.puzzle_dims):
                    if debug:
                        print(self.all_piece_dims, '12')
                    return False

            if min(width, height) > min(self.puzzle_dims):
                    if debug:
                        print(self.all_piece_dims, '13')
                    return False

            # check each edge and find the flat one(s)
            for i, edge in enumerate(piece2.edges):
                if edge.label == 'flat':
                    orientation = (i - piece2_edge_up) % 4
                    if orientation == 0: # up
                        # first, looking at the top of the puzzle. If there is one, side goes up, it must line up
                        if not self.top_edge is None:
                            if piece2_loc[1] != self.top_edge:
                                if debug:
                                    print(self.all_piece_dims, '14')
                                return False
                        elif piece2_loc[1] >= min_y: # if there is not a top side yet, this can't be at the min found so far
                            if debug:
                                print(self.all_piece_dims, '15')
                            return False
                        # if the bottom edge exists, 
                        if not self.bottom_edge is None:
                            # the height will be from this piece to the bottom edge
                            new_height = abs(self.bottom_edge - piece2_loc[1]) + 1
                            # if the side edges both exist, this new height must fit the defined dimensions
                            if not self.right_edge is None and not self.left_edge is None:
                                edge_width = self.right_edge - self.left_edge + 1
                                if edge_width == max(self.puzzle_dims):
                                    if new_height != min(self.puzzle_dims):
                                        if debug:
                                            print(self.all_piece_dims, '16')
                                        return False
                                else:
                                    if new_height != max(self.puzzle_dims):
                                        if debug:
                                            print(self.all_piece_dims, '17')
                                        return False
                            elif not new_height in self.puzzle_dims: # otherwise, the height should at least be in the puzzle dimensions
                                if debug:
                                    print(self.all_piece_dims, '18')
                                return False
                        # not for corner pieces
                        if piece2.type == 'side':
                            # side pieces pointing up can't be on the left / right side of the puzzle. Corner pieces go there.
                            if self.right_edge == piece2_loc[0]:
                                if debug:
                                    print(self.all_piece_dims, '19')
                                return False
                            elif self.right_edge == piece2_loc[0] + max(self.puzzle_dims) - 1:
                                if debug:
                                    print(self.all_piece_dims, '20')
                                return False
                            elif self.right_edge == piece2_loc[0] + min(self.puzzle_dims) - 1:
                                if height > min(self.puzzle_dims) or middle_height > min(self.puzzle_dims) - 2:
                                    if debug:
                                        print(self.all_piece_dims, '21')
                                    return False

                            if self.left_edge == piece2_loc[0]:
                                if debug:
                                    print(self.all_piece_dims, '22')
                                return False
                            elif self.left_edge == piece2_loc[0] - (max(self.puzzle_dims) - 1):
                                if debug:
                                    print(self.all_piece_dims, '23')
                                return False
                            elif self.left_edge == piece2_loc[0] - (min(self.puzzle_dims) - 1):
                                if height > min(self.puzzle_dims) or middle_height > min(self.puzzle_dims) - 2:
                                    if debug:
                                        print(self.all_piece_dims, '24')
                                    return False

                            # if neither side edge has been defined, but this piece extends the width in a bad way
                            if self.right_edge is None and self.left_edge is None:
                                if width > max(self.puzzle_dims) - 2:
                                    if debug:
                                        print(self.all_piece_dims, '25')
                                    return False
                                if height > max(self.puzzle_dims) - 2 and width > min(self.puzzle_dims) - 2:
                                    if debug:
                                        print(self.all_piece_dims, '26')
                                    return False

                    elif orientation == 1: # same for up but right
                        if not self.right_edge is None:
                            if piece2_loc[0] != self.right_edge:
                                if debug:
                                    print(self.all_piece_dims, '27')
                                return False
                        elif piece2_loc[0] <= max_x:
                            if debug:
                                print(self.all_piece_dims, '28')
                            return False
                        if not self.left_edge is None:
                            new_width = abs(piece2_loc[0] - self.left_edge) + 1
                            if not self.bottom_edge is None and not self.top_edge is None:
                                edge_height = self.bottom_edge - self.top_edge + 1
                                if edge_height == max(self.puzzle_dims):
                                    if new_width != min(self.puzzle_dims):
                                        if debug:
                                            print(self.all_piece_dims, '29')
                                        return False
                                else:
                                    if new_width != max(self.puzzle_dims):
                                        if debug:
                                            print(self.all_piece_dims, '30')
                                        return False
                            elif not new_width in self.puzzle_dims:
                                if debug:
                                    print(self.all_piece_dims, '31')
                                return False
                        if piece2.type == 'side':
                            if self.bottom_edge == piece2_loc[1]:
                                if debug:
                                    print(self.all_piece_dims, '32')
                                return False
                            elif self.bottom_edge == piece2_loc[1] + max(self.puzzle_dims) - 1:
                                if debug:
                                    print(self.all_piece_dims, '33')
                                return False
                            elif self.bottom_edge == piece2_loc[1] + min(self.puzzle_dims) - 1:
                                if width > min(self.puzzle_dims) or middle_width > min(self.puzzle_dims) - 2:
                                    if debug:
                                        print(self.all_piece_dims, '34')
                                    return False
                            if self.top_edge == piece2_loc[1]:
                                if debug:
                                    print(self.all_piece_dims, '35')
                                return False
                            elif self.top_edge == piece2_loc[1] - (max(self.puzzle_dims) - 1):
                                if debug:
                                    print(self.all_piece_dims, '36')
                                return False
                            elif self.top_edge == piece2_loc[1] - (min(self.puzzle_dims) - 1):
                                if width > min(self.puzzle_dims) or middle_width > min(self.puzzle_dims) - 2:
                                    if debug:
                                        print(self.all_piece_dims, '37')
                                    return False

                            if self.top_edge is None and self.bottom_edge is None:
                                if height > max(self.puzzle_dims) - 2:
                                    if debug:
                                        print(self.all_piece_dims, '38')
                                    return False
                                if width > max(self.puzzle_dims) - 2 and height > min(self.puzzle_dims) - 2:
                                    if debug:
                                        print(self.all_piece_dims, '39')
                                    return False
                                    
                    elif orientation == 2: # same for up but down
                        if not self.bottom_edge is None:
                            if piece2_loc[1] != self.bottom_edge:
                                if debug:
                                    print(self.all_piece_dims, '40')
                                return False
                        elif piece2_loc[1] <= max_y:
                            if debug:
                                print(self.all_piece_dims, '41')
                            return False
                        if not self.top_edge is None:
                            new_height = abs(piece2_loc[1] - self.top_edge) + 1
                            if not self.right_edge is None and not self.left_edge is None:
                                edge_width = self.right_edge - self.left_edge + 1
                                if edge_width == max(self.puzzle_dims):
                                    if new_height != min(self.puzzle_dims):
                                        if debug:
                                            print(self.all_piece_dims, '42')
                                        return False
                                else:
                                    if new_height != max(self.puzzle_dims):
                                        if debug:
                                            print(self.all_piece_dims, '43')
                                        return False
                            elif not new_height in self.puzzle_dims:
                                if debug:
                                    print(self.all_piece_dims, '44')
                                return False
                        if piece2.type == 'side':
                            if self.right_edge == piece2_loc[0]:
                                if debug:
                                    print(self.all_piece_dims, '45')
                                return False
                            elif self.right_edge == piece2_loc[0] + max(self.puzzle_dims) - 1:
                                if debug:
                                    print(self.all_piece_dims, '46')
                                return False
                            elif self.right_edge == piece2_loc[0] + min(self.puzzle_dims) - 1:
                                if height > min(self.puzzle_dims) or middle_height > min(self.puzzle_dims) - 2:
                                    if debug:
                                        print(self.all_piece_dims, '47')
                                    return False

                            if self.left_edge == piece2_loc[0]:
                                if debug:
                                    print(self.all_piece_dims, '48')
                                return False
                            elif self.left_edge == piece2_loc[0] - (max(self.puzzle_dims) - 1):
                                if debug:
                                    print(self.all_piece_dims, '49')
                                return False
                            elif self.left_edge == piece2_loc[0] - (min(self.puzzle_dims) - 1):
                                if height > min(self.puzzle_dims) or middle_height > min(self.puzzle_dims) - 2:
                                    if debug:
                                        print(self.all_piece_dims, '50')
                                    return False

                            if self.right_edge is None and self.left_edge is None:
                                if width > max(self.puzzle_dims) - 2:
                                    if debug:
                                        print(self.all_piece_dims, '51')
                                    return False
                                if height > max(self.puzzle_dims) - 2 and width > min(self.puzzle_dims) - 2:
                                    if debug:
                                        print(self.all_piece_dims, '52')
                                    return False
                    else: # same as for up but left
                        if not self.left_edge is None:
                            if piece2_loc[0] != self.left_edge:
                                if debug:
                                    print(self.all_piece_dims, '53')
                                return False
                        elif piece2_loc[0] >= min_x:
                            if debug:
                                print(self.all_piece_dims, '54')
                            return False
                        if not self.right_edge is None:
                            new_width = abs(self.right_edge - piece2_loc[0]) + 1
                            if not self.bottom_edge is None and not self.top_edge is None:
                                edge_height = self.bottom_edge - self.top_edge + 1
                                if edge_height == max(self.puzzle_dims):
                                    if new_width != min(self.puzzle_dims):
                                        if debug:
                                            print(self.all_piece_dims, '55')
                                        return False
                                else:
                                    if new_width != max(self.puzzle_dims):
                                        if debug:
                                            print(self.all_piece_dims, '56')
                                        return False
                            elif not new_width in self.puzzle_dims:
                                if debug:
                                    print(self.all_piece_dims, '57')
                                return False
                        if piece2.type == 'side':
                            if self.bottom_edge == piece2_loc[1]:
                                if debug:
                                    print(self.all_piece_dims, '58')
                                return False
                            elif self.bottom_edge == piece2_loc[1] + max(self.puzzle_dims) - 1:
                                if debug:
                                    print(self.all_piece_dims, '59')
                                return False
                            elif self.bottom_edge == piece2_loc[1] + min(self.puzzle_dims) - 1:
                                if width > min(self.puzzle_dims) or middle_width > min(self.puzzle_dims) - 2:
                                    if debug:
                                        print(self.all_piece_dims, '60')
                                    return False

                            if self.top_edge == piece2_loc[1]:
                                if debug:
                                    print(self.all_piece_dims, '61')
                                return False
                            elif self.top_edge == piece2_loc[1] - (max(self.puzzle_dims) - 1):
                                if debug:
                                    print(self.all_piece_dims, '62')
                                return False
                            elif self.top_edge == piece2_loc[1] - (min(self.puzzle_dims) - 1):
                                if width > min(self.puzzle_dims) or middle_width > min(self.puzzle_dims) - 2:
                                    if debug:
                                        print(self.all_piece_dims, '63')
                                    return False

                            if self.top_edge is None and self.bottom_edge is None:
                                if height > max(self.puzzle_dims) - 2:
                                    if debug:
                                        print(self.all_piece_dims, '64')
                                    return False
                                if width > max(self.puzzle_dims) - 2 and height > min(self.puzzle_dims) - 2:
                                    if debug:
                                        print(self.all_piece_dims, '65')
                                    return False
        return True # if none of the above conditions were a thing, return true

    def splicePartialSolutionImages(self, solution_image, img1, img2, corners1, corners2, midpoint, direction, c2_in_solution=False, with_details=False, desired_corner=None, resize_factor=0.8):

        # assumed that corners1 coords < corners2 in the final image along one of the axis
        entry = self.position_dict.get(midpoint)
        if entry is None:
            return solution_image, corners1, corners2

        piece, edge_up = entry
        
        if direction % 2 == 0: # either down or up
            e1 = 1 # edge on c1
            e2 = 3 # edge on c2
        else:
            e1 = 2
            e2 = 0
        if direction == 1 or direction == 2:
            init_pt = -1
        else:
            init_pt = 0
        if direction == 0:
            ci1 = 1
            ci2 = 0
            ci12 = 2
            ci22 = 3
            piece_indices = []
            for i in range(len(corners1[e1])):
                piece_indices.append((midpoint[0], midpoint[1] + i))
        elif direction == 1:
            ci1 = 2
            ci2 = 1
            ci12 = 3
            ci22 = 0
            piece_indices = []
            for i in range(len(corners1[e1])):
                piece_indices.append((midpoint[0] - i, midpoint[1]))
        elif direction == 2:
            ci1 = 2
            ci2 = 3
            ci12 = 1
            ci22 = 0
            piece_indices = []
            for i in range(len(corners1[e1])):
                piece_indices.append((midpoint[0], midpoint[1] - i))
        else:
            ci1 = 3
            ci2 = 0
            ci12 = 2
            ci22 = 1
            piece_indices = []
            for i in range(len(corners1[e1])):
                piece_indices.append((midpoint[0] + i, midpoint[1]))


        if not c2_in_solution:
            flat_edges = [i for i in range(4) if piece.edges[i].label == 'flat']
            if len(flat_edges) > 0:
                flat_edge = max(flat_edges)
                rel_edge = (flat_edge - edge_up) % 4
            else:
                rel_edge = 0

            piece_image, piece_corner_poses = piece.getSubimage2(edge_up, with_details=with_details, resize_factor=resize_factor, rel_edge=rel_edge)
            ph, pw, _ = img2.shape

            if not desired_corner is None:
                pih, piw, _ = piece_image.shape
                piece_width = piece_corner_poses[1,0] - piece_corner_poses[0,0]
                piece_left_corner = desired_corner[0] - (corners2[2][-1][2][0] - corners2[2][0][3][0])
                piece_new_width = piece_left_corner - corners1[1][-1][2][0]
                scale = piece_new_width / piece_width
                if scale * piw > 0:
                    piece_corner_poses[:,0] = (piece_corner_poses[:,0] * (scale)).astype(int)
                    piece_image = cv2.resize(piece_image, (int(scale * piw), pih), interpolation=cv2.INTER_AREA)
                new_coord = corners1[1][-1][2] + np.array([piece_new_width, 0]) - corners2[3][-1][3]
            else:
                new_coord = corners1[e1][init_pt][ci1] + (piece_corner_poses[ci1] - piece_corner_poses[ci2]) - corners2[e2][init_pt][ci2]
            
            for i, edge in enumerate(corners2):
                if len(corners2[i]) > 0:
                    corners2[i] = corners2[i] + new_coord

            x_coord = int(new_coord[0])
            y_coord = int(new_coord[1])


            mask = ((np.sum(img2, axis=2) > 0) * 255).astype(np.uint8)
            solution_image_section = solution_image[y_coord:y_coord + ph, x_coord:x_coord + pw]
            if solution_image_section.shape[:2] == mask.shape:
                solution_image_section[mask == 255] = np.array([0,0,0])
                solution_image[y_coord:y_coord + ph, x_coord:x_coord + pw] = cv2.bitwise_xor(solution_image_section, img2)
            

            # for pt in corners2[e2]:
            #     cv2.circle(solution_image, (int(pt[ci2][0]), int(pt[ci2][1])), 15, (0,255,255), -1)

        for pt_index, pt in enumerate(piece_indices):
            piece, edge_up = self.position_dict.get(pt)

            if direction == 1 or direction == 2:
                index = len(corners1[e1]) - (pt_index + 1)
            else:
                index = pt_index

            flat_edges = [i for i in range(4) if piece.edges[i].label == 'flat']
            if len(flat_edges) > 0:
                flat_edge = max(flat_edges)
                rel_edge = (flat_edge - edge_up) % 4
            else:
                rel_edge = 0

            piece_image, piece_corner_poses = piece.getSubimage2(edge_up, with_details=with_details, resize_factor=resize_factor, rel_edge=rel_edge)

            ph, pw, _ = piece_image.shape

            src_piece_corners = piece_corner_poses.astype(np.float32)
            desired_piece_corners = (piece_corner_poses + np.array([pw/2, ph/2])).astype(np.float32)
            desired_piece_corners[ci1] = desired_piece_corners[ci2] + (corners2[e2][index][ci2] - corners1[e1][index][ci1])
            desired_piece_corners[ci12] = desired_piece_corners[ci1] + (corners2[e2][index][ci22] - corners2[e2][index][ci2])
            desired_piece_corners[ci22] = desired_piece_corners[ci2] + (corners1[e1][index][ci12] - corners1[e1][index][ci1])

            transform = cv2.getPerspectiveTransform(src_piece_corners, desired_piece_corners)
            img_warped = cv2.warpPerspective(piece_image, transform, (2*pw,2*ph))

            ph, pw, _ = img_warped.shape

            new_coord = corners1[e1][index][ci1] - desired_piece_corners[ci2]
            x_coord = int(new_coord[0])
            y_coord = int(new_coord[1])

            mask = ((np.sum(img_warped, axis=2) > 0) * 255).astype(np.uint8)
            
            pwt = piece_corner_poses[1][0] - piece_corner_poses[0][0]
            phl = piece_corner_poses[3][1] - piece_corner_poses[0][1]
            pwb = piece_corner_poses[2][0] - piece_corner_poses[3][0]
            phr = piece_corner_poses[2][1] - piece_corner_poses[1][1]

            top_diff = desired_piece_corners[1][0] - desired_piece_corners[0][0]
            left_diff = desired_piece_corners[3][1] - desired_piece_corners[0][1]
            bottom_diff = desired_piece_corners[2][0] - desired_piece_corners[3][0]
            right_diff = desired_piece_corners[2][1] - desired_piece_corners[1][1]
            
            if not (top_diff <= pwt/10 or left_diff <= phl/10 or bottom_diff <= pwb/10 or right_diff <= phr/10):  
                solution_image_section = solution_image[y_coord:y_coord + ph, x_coord:x_coord + pw]
                if solution_image_section.shape[:2] == mask.shape:
                    solution_image_section[mask == 255] = np.array([0,0,0])
                    solution_image[y_coord:y_coord + ph, x_coord:x_coord + pw] = cv2.bitwise_xor(solution_image_section, img_warped)
            
            # cv2.imshow('sol', cv2.resize(solution_image, (800,800), interpolation=cv2.INTER_AREA))
            # cv2.waitKey(0)

        return solution_image, corners1, corners2

    # returns an image containing the solution to the puzzle
    def getSolutionImage(self, with_details=False, draw_edges=[], resize_factor=0.8, just_sides=False):
        # get the bounds of the solution
        keys = self.position_dict.keys()
        min_x = min(keys, key=lambda x:x[0])[0]
        max_x = max(keys, key=lambda x:x[0])[0]

        min_y = min(keys, key=lambda x:x[1])[1]
        max_y = max(keys, key=lambda x:x[1])[1]

        # get the subimages for each piece in the solution facing the appropriate direction
        # image_dict = {}
        max_size = 0
        for key in keys:
            piece, edge_up = self.position_dict[key]
            # draw_edges_pc = [edge[1] for edge in draw_edges if edge[0] == piece] 
            piece_image, _ = piece.getSubimage2(edge_up, with_details=with_details, resize_factor=resize_factor)
            # image_dict[key] = piece_image

            piece_image_size = max(piece_image.shape)
            if piece_image_size > max_size:
                max_size = piece_image_size

        # add them all to the solution image
        w = max_x - min_x + 1
        h = max_y - min_y + 1

        if just_sides:
            num_sections_x = 2
            num_sections_y = 2
        else:
            num_sections_x = max(2, (w // 9))
            num_sections_y = max(2, (h // 9))

        x_midpoints = np.around(np.linspace(min_x - 1, max_x + 1, num_sections_x + 1)).astype(int)
        y_midpoints = np.around(np.linspace(min_y - 1, max_y + 1, num_sections_y + 1)).astype(int)

        imgs = [[None for y in range(num_sections_y)] for x in range(num_sections_x)]
        corners = [[None for y in range(num_sections_y)] for x in range(num_sections_x)]

        for x in range(num_sections_x):
            for y in range(num_sections_y):
                if y == 0:
                    direction = 0
                elif y == num_sections_y - 1:
                    direction = 2
                elif x == 0:
                    direction = 3
                else:
                    direction = 0
                img_xy, corners_xy = self.getPartialSolutionImage(x_midpoints[x] + 1, 
                                    x_midpoints[x+1] - 1, y_midpoints[y] + 1, 
                                    y_midpoints[y+1] - 1, max_size, direction, resize_factor=resize_factor)
                h_xy, w_xy, _ = img_xy.shape
                
                if min(len(edge) for edge in corners_xy) > 1:
                    x_ratio = 1
                    y_ratio = 1
                    if y != 0 and len(corners[x][y-1][2]) > 0 and len(corners_xy[0]) > 0:
                        c1 = corners_xy[0][0][0]
                        c2 = corners_xy[0][-1][1]
                        c1_prev = corners[x][y-1][2][0][3]
                        c2_prev = corners[x][y-1][2][-1][2]
                        x_ratio = (c2_prev[0] - c1_prev[0]) / (c2[0] - c1[0])
                    if x != 0 and len(corners[x-1][y][1]) > 0 and len(corners_xy[3]) > 0:
                        c1 = corners_xy[3][0][0]
                        c3 = corners_xy[3][-1][3]
                        c1_prev = corners[x-1][y][1][0][1]
                        c2_prev = corners[x-1][y][1][-1][2]
                        y_ratio = (c2_prev[1] - c1_prev[1]) / (c3[1] - c1[1])
                    
                    if x_ratio != 1 or y_ratio != 1:
                        img_xy = cv2.resize(img_xy, (int(w_xy * x_ratio), int(h_xy * y_ratio)), interpolation=cv2.INTER_AREA)
                        for i, edge in enumerate(corners_xy):
                            edge[:,:,0] = (edge[:,:,0] * x_ratio).astype(int)
                            edge[:,:,1] = (edge[:,:,1] * y_ratio).astype(int)
                            corners_xy[i] = edge

                imgs[x][y] = img_xy
                corners[x][y] = corners_xy

        solution_image = np.zeros((max_size * (h + 6), max_size * (w + 6), 3), dtype=np.uint8)
        hs, ws, _ = solution_image.shape
        h1, w1, _ = imgs[0][0].shape
        solution_image[3 * max_size:3 * max_size + h1, 3 * max_size:3 * max_size + w1] = imgs[0][0]

        for i, edge in enumerate(corners[0][0]):
            if len(corners[0][0][i]) != 0:
                corners[0][0][i] = edge + np.array([3 * max_size, 3 * max_size])

        x_midpoints[0] += 1
        x_midpoints[-1] -= 1

        y_midpoints[0] += 1
        y_midpoints[-1] -= 1


        for y in range(num_sections_y):
            for x in range(num_sections_x):
                if x + 1 < num_sections_x: # go right
                    if y == num_sections_y - 1:
                        direction = 2
                        midpoint = (x_midpoints[x+1], y_midpoints[-1])
                    elif y == 0:
                        direction = 0
                        midpoint = (x_midpoints[x+1], y_midpoints[0])
                    else:
                        direction = 0
                        midpoint = (x_midpoints[x+1], y_midpoints[y]+1)

                    if y > 0 and len(corners[x+1][y][2]) > 0 and len(corners[x+1][y-1][2]) > 0 and len(corners[x][y][2]) > 0:
                        desired_corner = (corners[x+1][y-1][2][-1][2][0], corners[x][y][2][-1][2][1])
                    else:
                        desired_corner = None

                    solution_image, new_c1, new_c2 = self.splicePartialSolutionImages(
                        solution_image, imgs[x][y], imgs[x+1][y], 
                        corners[x][y], corners[x+1][y], midpoint, 
                        direction, c2_in_solution=False, desired_corner=desired_corner, resize_factor=resize_factor)
                    corners[x][y] = new_c1
                    corners[x+1][y] = new_c2

                    if y > 0:
                        if x == num_sections_x - 2:
                            direction = 1
                            midpoint = (x_midpoints[x+2], y_midpoints[y])
                        else:
                            direction = 3
                            midpoint = (x_midpoints[x+1] + 1, y_midpoints[y])

                        solution_image, new_c1, new_c2 = self.splicePartialSolutionImages(
                            solution_image, imgs[x+1][y-1], imgs[x+1][y], 
                            corners[x+1][y-1], corners[x+1][y], midpoint, 
                            direction, c2_in_solution=True, resize_factor=resize_factor)
                        corners[x+1][y-1] = new_c1
                        corners[x+1][y] = new_c2

                        middle_piece_loc = (x_midpoints[x+1], y_midpoints[y])
                        entry = self.position_dict.get(middle_piece_loc)
                        if entry is None:
                            continue
                        middle_piece, edge_up = entry

                        piece_image, piece_corner_poses = middle_piece.getSubimage2(edge_up, with_details=with_details, resize_factor=resize_factor)

                        ph, pw, _ = piece_image.shape

                        src_piece_corners = piece_corner_poses.astype(np.float32)
                        desired_piece_corners = (piece_corner_poses + np.array([pw/2, ph/2])).astype(np.float32)
                        desired_piece_corners[1] = desired_piece_corners[0] + (corners[x+1][y-1][2][0][3] - corners[x][y-1][2][-1][2])
                        desired_piece_corners[2] = desired_piece_corners[1] + (corners[x+1][y][0][0][0] - corners[x+1][y-1][2][0][3])
                        desired_piece_corners[3] = desired_piece_corners[0] + (corners[x][y][0][-1][1] - corners[x][y-1][2][-1][2])

                        transform = cv2.getPerspectiveTransform(src_piece_corners, desired_piece_corners)
                        img_warped = cv2.warpPerspective(piece_image, transform, (ph*2,pw*2))

                        coord = (corners[x][y-1][2][-1][2] - desired_piece_corners[0]).astype(int)
                        x_coord = coord[0]
                        y_coord = coord[1]

                        ph, pw, _ = img_warped.shape

                        pwt = piece_corner_poses[1][0] - piece_corner_poses[0][0]
                        phl = piece_corner_poses[3][1] - piece_corner_poses[0][1]
                        pwb = piece_corner_poses[2][0] - piece_corner_poses[3][0]
                        phr = piece_corner_poses[2][1] - piece_corner_poses[1][1]

                        top_diff = desired_piece_corners[1][0] - desired_piece_corners[0][0]
                        left_diff = desired_piece_corners[3][1] - desired_piece_corners[0][1]
                        bottom_diff = desired_piece_corners[2][0] - desired_piece_corners[3][0]
                        right_diff = desired_piece_corners[2][1] - desired_piece_corners[1][1]
                        
                        if not (top_diff <= pwt/10 or left_diff <= phl/10 or bottom_diff <= pwb/10 or right_diff <= phr/10):
                            mask = ((np.sum(img_warped, axis=2) > 0) * 255).astype(np.uint8)
                            solution_image_section = solution_image[y_coord:y_coord + ph, x_coord:x_coord + pw]
                            if solution_image_section.shape[:2] == mask.shape:
                                solution_image_section[mask == 255] = np.array([0,0,0])
                                solution_image[y_coord:y_coord + ph, x_coord:x_coord + pw] = cv2.bitwise_xor(solution_image_section, img_warped)
                        

            if y + 1 < num_sections_y: # go down
                solution_image, new_c1, new_c2 = self.splicePartialSolutionImages(
                    solution_image, imgs[0][y], imgs[0][y+1], 
                    corners[0][y], corners[0][y+1], (x_midpoints[0], y_midpoints[y+1]), 
                    3, c2_in_solution=False, resize_factor=resize_factor)
                corners[0][y] = new_c1
                corners[0][y+1] = new_c2


        hs, hw, _ = solution_image.shape
        solution_image_crop = solution_image.copy()

        zero_cols = np.argwhere(np.sum(solution_image, axis=0) == 0)
        solution_image_crop = np.delete(solution_image_crop, zero_cols, axis=1)

        zero_rows = np.argwhere(np.sum(solution_image, axis=1) == 0)
        solution_image_crop = np.delete(solution_image_crop, zero_rows, axis=0)

        color_sums = np.sum(solution_image_crop, axis=2)
        col_delete_index = 0
        for col_sum in np.sum(color_sums == 0, axis=0):
            if col_sum > 0.2 * hs:
                break
            col_delete_index += 1
        solution_image_crop = solution_image_crop[col_delete_index:]

        color_sums = np.sum(solution_image_crop, axis=2)
        col_delete_index = 0
        for col_sum in np.sum(color_sums == 0, axis=0)[::-1]:
            if col_sum > 0.2 * hs:
                break
            col_delete_index += 1
        solution_image_crop = solution_image_crop[:-(col_delete_index+1)]

        color_sums = np.sum(solution_image_crop, axis=2)
        row_delete_index = 0
        for row_sum in np.sum(color_sums == 0, axis=1):
            if row_sum > 0.2 * hw:
                break
            row_delete_index += 1
        solution_image_crop = solution_image_crop[:, row_delete_index:]

        color_sums = np.sum(solution_image_crop, axis=2)
        row_delete_index = 0
        for row_sum in np.sum(color_sums == 0, axis=1)[::-1]:
            if row_sum > 0.2 * hw:
                break
            row_delete_index += 1
        solution_image_crop = solution_image_crop[:, :-(row_delete_index+1)]

        if solution_image_crop is None:
            return solution_image
        else:
            return solution_image_crop

    def getPartialSolutionImage(self, min_x, max_x, min_y, max_y, max_size, direction, with_details=False, draw_edges=[], resize_factor=0.8):

        # add them all to the solution image
        w = max_x - min_x
        h = max_y - min_y
        solution_image = np.zeros((max_size * (h + 6), max_size * (w + 6), 3), dtype=np.uint8)
        hs, ws, _ = solution_image.shape

        if direction == 0:
            corner_order = [0, 1, 2, 3]
            x_range = range(min_x, max_x + 1)
            y_range = range(min_y, max_y + 1)
            init_pos = (3 * max_size, 3 * max_size)
        elif direction == 1:
            corner_order = [1, 0, 3, 2]
            x_range = range(max_x, min_x - 1, -1)
            y_range = range(min_y, max_y + 1)
            init_pos = (ws - 3 * max_size, 3 * max_size)
        elif direction == 2:
            corner_order = [2, 3, 0, 1]
            x_range = range(max_x, min_x - 1, -1)
            y_range = range(max_y, min_y - 1, -1)
            init_pos = (ws - 3 * max_size, hs - 3 * max_size)
        else:
            corner_order = [3, 2, 1, 0]
            x_range = range(min_x, max_x + 1)
            y_range = range(max_y, min_y - 1, -1)
            init_pos = (3 * max_size, hs - 3 * max_size)
           
        edge_corner_poses = [[], [], [], []]

        left_corner_poses = {y:None for y in range(min_y, max_y+1)}
        curr_left_corner_poses = {y:None for y in range(min_y, max_y+1)}
        for x in x_range:
            above_corners = None
            left_corner_poses = curr_left_corner_poses.copy()
            curr_left_corner_poses = {y:None for y in range(min_y, max_y+1)}
            for y in y_range:
                entry = self.position_dict.get((x, y))
                if entry is None:
                    above_corners = None
                    continue
                piece, edge_up = entry
                flat_edges = [i for i in range(4) if piece.edges[i].label == 'flat']
                add_angle = 0
                if len(flat_edges) > 0:
                    flat_edge = max(flat_edges)
                    # add_angle = (edge_up - flat_edge) % 4 * 90
                    rel_edge = (flat_edge - edge_up) % 4
                else:
                    rel_edge = 0

                draw_edges_pc = [edge[1] for edge in draw_edges if edge[0] == piece]

                piece_image, corner_poses = piece.getSubimage2(edge_up, with_details=with_details, resize_factor=resize_factor, draw_edges=draw_edges_pc, rel_edge=rel_edge, line_width=4)

                left_corners = left_corner_poses[y]

                ph, pw, _ = piece_image.shape

                corner_poses = corner_poses.astype(np.float32)
                desired_corner_poses = (corner_poses + np.array([pw/2, ph/2])).astype(np.float32)

                
                if not (x == min_x and len(flat_edges) > 0 and rel_edge == 3):
                    if not left_corners is None:
                        # desired_corner_poses[0] = corner_poses[0]
                        desired_corner_poses[corner_order[3]] = desired_corner_poses[corner_order[0]] + (left_corners[corner_order[2]] - left_corners[corner_order[1]])

                if not (y == min_y and len(flat_edges) > 0 and rel_edge == 0):
                    if not above_corners is None:
                        desired_corner_poses[corner_order[1]] = desired_corner_poses[corner_order[0]] + (above_corners[corner_order[2]] - above_corners[corner_order[3]])
                

                if len(flat_edges) == 2:
                    desired_corner_poses[corner_order[1]] = np.array([desired_corner_poses[corner_order[1]][0], desired_corner_poses[corner_order[0]][1]])
                    desired_corner_poses[corner_order[3]] = np.array([desired_corner_poses[corner_order[0]][0], desired_corner_poses[corner_order[3]][1]])
                    desired_corner_poses[corner_order[2]] = np.array([desired_corner_poses[corner_order[1]][0], desired_corner_poses[corner_order[3]][1]])
                elif len(flat_edges) == 1:
                    if x == max_x and rel_edge == 1:
                        desired_corner_poses[corner_order[2]] = desired_corner_poses[corner_order[1]] + (corner_poses[corner_order[2]] - corner_poses[corner_order[1]])
                    elif y == max_y and rel_edge == 2:
                        desired_corner_poses[corner_order[2]] = desired_corner_poses[corner_order[3]] + (corner_poses[corner_order[2]] - corner_poses[corner_order[3]])
                else:
                    desired_corner_poses[corner_order[2]] = corner_poses[corner_order[2]] + (desired_corner_poses[corner_order[1]] - corner_poses[corner_order[3]] + desired_corner_poses[corner_order[3]] - corner_poses[corner_order[1]]) / 2
                
                transform = cv2.getPerspectiveTransform(corner_poses, desired_corner_poses)
                img_warped = cv2.warpPerspective(piece_image, transform, (2*pw,2*ph))


                pwt = corner_poses[1][0] - corner_poses[0][0]
                phl = corner_poses[3][1] - corner_poses[0][1]
                pwb = corner_poses[2][0] - corner_poses[3][0]
                phr = corner_poses[2][1] - corner_poses[1][1]

                corner_poses = desired_corner_poses.astype(int)
                # cv2.imshow("img_warp", img_warped)

                ph, pw, _ = img_warped.shape 

                if left_corners is None:
                    if above_corners is None:
                        x_coord, y_coord = init_pos
                    else:
                        x_coord = above_corners[corner_order[3]][0] - corner_poses[corner_order[0]][0]
                        y_coord = above_corners[corner_order[3]][1] - corner_poses[corner_order[0]][1]
                else:
                    x_coord = left_corners[corner_order[1]][0] - corner_poses[corner_order[0]][0]
                    y_coord = left_corners[corner_order[1]][1] - corner_poses[corner_order[0]][1]


                top_diff = desired_corner_poses[1][0] - desired_corner_poses[0][0]
                left_diff = desired_corner_poses[3][1] - desired_corner_poses[0][1]
                bottom_diff = desired_corner_poses[2][0] - desired_corner_poses[3][0]
                right_diff = desired_corner_poses[2][1] - desired_corner_poses[1][1]
                
                if not (top_diff <= pwt/10 or left_diff <= phl/10 or bottom_diff <= pwb/10 or right_diff <= phr/10):
                    mask = ((np.sum(img_warped, axis=2) > 0) * 255).astype(np.uint8)
                    solution_image_section = solution_image[y_coord:y_coord + ph, x_coord:x_coord + pw]
                    if solution_image_section.shape[:2] == mask.shape:
                        solution_image_section[mask == 255] = np.array([0,0,0])
                        solution_image[y_coord:y_coord + ph, x_coord:x_coord + pw] = cv2.bitwise_xor(solution_image_section, img_warped)

                # cv2.imshow('sol', cv2.resize(solution_image, (800,800), interpolation=cv2.INTER_AREA))
                # cv2.waitKey(1)

                curr_left_corner_poses[y] = corner_poses + np.array([x_coord, y_coord])
                above_corners = corner_poses + np.array([x_coord, y_coord])

                if y == min_y:
                    edge_corner_poses[0].append(above_corners)
                elif y == max_y:
                    edge_corner_poses[2].append(above_corners)
                if x == min_x:
                    edge_corner_poses[3].append(above_corners)
                elif x == max_x:
                    edge_corner_poses[1].append(above_corners)

        zero_cols = np.argwhere(np.sum(solution_image, axis=0) == 0)
        first_col = 0
        for i, sum_val in enumerate(np.sum(solution_image, axis=0)):
            if np.sum(sum_val) != 0:
                first_col = i-1
                break
        solution_image = np.delete(solution_image, zero_cols, axis=1)

        zero_rows = np.argwhere(np.sum(solution_image, axis=1) == 0)
        first_row = 0
        for i, sum_val in enumerate(np.sum(solution_image, axis=1)):
            if np.sum(sum_val) != 0:
                first_row = i-1
                break
        
        solution_image = np.delete(solution_image, zero_rows, axis=0)
        for i, edge in enumerate(edge_corner_poses):
            if len(edge) > 0:
                edge_corner_poses[i] = edge - np.array([first_col, first_row])

        edge_corner_poses[0] = np.array(sorted(edge_corner_poses[0], key=lambda x:x[0][0]))
        edge_corner_poses[1] = np.array(sorted(edge_corner_poses[1], key=lambda x:x[0][1]))
        edge_corner_poses[2] = np.array(sorted(edge_corner_poses[2], key=lambda x:x[0][0]))
        edge_corner_poses[3] = np.array(sorted(edge_corner_poses[3], key=lambda x:x[0][1]))

        return solution_image, edge_corner_poses

# returns the location and edge facing up of piece2 with edge edge2 connnecting to edge1 on piece1
def getPieceInfo(piece1, edge1, edge2, piece1_loc, piece1_edge_up):
    # up, right, down, left
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
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


def getConnectedComponents(edges):
    edge_dict = {}
    pieces = []
    for edge in edges:
        edge_dict[(edge[0], edge[1])] = (edge[2], edge[3])

    pieces = [edge[0] for edge in edges]
    unvisited_pieces = set(pieces)
    num_pieces = len(unvisited_pieces)
    connected_components = []
    while len(unvisited_pieces) > 0:
        num_connected, connected_set = dfs(edge_dict, unvisited_pieces, random.choice(list(unvisited_pieces)))
        connected_components.append(connected_set)
    # print(len(largest_section), num_pieces)
    connected_components = sorted(connected_components, key=lambda x:-len(x))
    return connected_components

class ListDict(object):
    def __init__(self):
        self.position_dict = {}
        self.list = []

    def __contains__(self, item):
        return (item in self.position_dict)

    def __len__(self):
        return len(self.list)

    def __iter__(self):
        return iter(self.list)
    
    def add(self, item):
        if item in self.position_dict:
            return
        if len(self.list) <= 1:
            self.list.append(item)
            self.position_dict[item] = len(self.list) - 1
        else:
            position = random.choice(range(len(self.list)))
            prev_item = self.list[position]
            self.list[position] = item
            self.list.append(prev_item)
            self.position_dict[item] = position
            self.position_dict[prev_item] = len(self.list)-1
    
    def remove(self, item):
        position = self.position_dict.pop(item)
        last = self.list.pop()
        if position != len(self.list):
            self.list[position] = last
            self.position_dict[last] = position

    def random_choice(self):
        return random.choice(self.list)


def dfs(edge_dict, unvisited_pieces, piece):
    num_connected = 0
    connected = set()
    if piece in unvisited_pieces:
        num_connected = 1
        connected.add(piece)
        unvisited_pieces.remove(piece)
        for piece_edge in range(4):
            connected_edge = edge_dict.get((piece, piece_edge))
            if connected_edge is None:
                continue
            piece2 = connected_edge[0]
            num_connected_piece2, connected_piece2 = dfs(edge_dict, unvisited_pieces, piece2)
            num_connected += num_connected_piece2
            connected = connected.union(connected_piece2)
    return num_connected, connected


if __name__=='__main__':
    main()

# so much code lol