from puzzleSolution import PuzzleSolution
from pieceCollection import PieceCollection
import random
import cv2
import numpy as np
from timeit import default_timer as timer
import math
import copy
import sys
import gc


def main():

    # puzzle_solver = PuzzleSolver("starwars", (6, 8), 30, 30,
    #                              [('input/star_wars_new.jpg', 48)], settings=[10, 50, 50, 5, 10, 64])

    # puzzle_solver = PuzzleSolver("pokemon_1", (14.7, 10.5), 100, 100,
    #     [('input/pokemon_puzzle_1_01.png', 20),
    #     ('input/pokemon_puzzle_1_02.png', 20),
    #     ('input/pokemon_puzzle_1_03.png', 20),
    #     ('input/pokemon_puzzle_1_04.png', 20),
    #     ('input/pokemon_puzzle_1_05.png', 20)], settings=[10, 50, 50, 14, 20, 32])

    # puzzle_solver = PuzzleSolver("pokemon_2", (15, 11), 100, 100,
    #     [('input/pokemon_puzzle_2_01.png', 20),
    #     ('input/pokemon_puzzle_2_02.png', 20),
    #     ('input/pokemon_puzzle_2_03.png', 20),
    #     ('input/pokemon_puzzle_2_04.png', 20),
    #     ('input/pokemon_puzzle_2_05.png', 20)], settings=[10, 50, 50, 14, 20, 32])

    # puzzle_solver = PuzzleSolver("300", (21.25, 15), 100, 100,
    #     [('input/300_01.png', 30), ('input/300_02.png', 30), ('input/300_03.png', 30), ('input/300_04.png', 30),
    #     ('input/300_05.png', 30), ('input/300_06.png', 30), ('input/300_07.png', 30), ('input/300_08.png', 30),
    #     ('input/300_09.png', 30), ('input/300_10.png', 30)], settings=[10, 25, 30, 8, 14, 32])

    # puzzle_solver = PuzzleSolver("tart4", (18, 18), 200, 100,
    #                              [('input/tart_puzzle_01.jpg', 30), ('input/tart_puzzle_02.jpg', 30), ('input/tart_puzzle_03.jpg', 30), ('input/tart_puzzle_04.jpg', 30),
    #                               ('input/tart_puzzle_05.jpg', 30), ('input/tart_puzzle_06.jpg',
    #                                                                  30), ('input/tart_puzzle_07.jpg', 28), ('input/tart_puzzle_08.jpg', 30),
    #                               ('input/tart_puzzle_09.jpg', 30), ('input/tart_puzzle_10.jpg', 30), ('input/tart_puzzle_11.jpg', 26)],
    #                              settings=[10, 50, 50, 8, 14, 64])

    # puzzle_solver = PuzzleSolver("feather11", (21.25, 15), 200, 500,
    #     [('input/feather2_01.jpg', 40),('input/feather2_02.jpg', 40),('input/feather2_03.jpg', 40),('input/feather2_04.jpg', 40),
    #     ('input/feather2_05.jpg', 40),('input/feather2_06.jpg', 40),('input/feather2_07.jpg', 40),('input/feather2_08.jpg', 20)],
    #     settings=[20, 40, 50, 12, 16, 32])

    # puzzle_solver = PuzzleSolver("travel", (21.25, 15), 200, 100,
    #     [('input/travel_puzzle_01.jpg', 30),('input/travel_puzzle_02.jpg', 30),('input/travel_puzzle_03.jpg', 30),('input/travel_puzzle_04.jpg', 30),
    #     ('input/travel_puzzle_05.jpg', 30),('input/travel_puzzle_06.jpg', 30),('input/travel_puzzle_07.jpg', 30),('input/travel_puzzle_08.jpg', 30),
    #     ('input/travel_puzzle_09.jpg', 30),('input/travel_puzzle_10.jpg', 12),('input/travel_puzzle_11.jpg', 18)],
    #      settings=[10, 50, 50, 10, 20, 64])

    # puzzle_solver = PuzzleSolver("owl3", (21.25, 15), 1000, 500,
    # [('input/owl2_01.jpg', 40), ('input/owl2_02.jpg', 40), ('input/owl2_03.jpg', 40),
    # ('input/owl2_04.jpg', 40), ('input/owl2_05.jpg', 40), ('input/owl2_06.jpg', 40),
    # ('input/owl2_07.jpg', 40), ('input/owl2_08.jpg', 20)], settings=[10, 30, 30, 12, 16, 64])

    # puzzle_solver = PuzzleSolver("shining", (18, 24), 10000, 10000,
    #     [('input/shining_01.jpg', 42), ('input/shining_02.jpg', 42), ('input/shining_03.jpg', 42), ('input/shining_04.jpg', 42),
    #     ('input/shining_05.jpg', 42), ('input/shining_06.jpg', 42), ('input/shining_07.jpg', 42), ('input/shining_08.jpg', 42),
    #     ('input/shining_09.jpg', 42), ('input/shining_10.jpg', 42), ('input/shining_11.jpg', 42), ('input/shining_12.jpg', 23),
    #     ('input/shining_13.jpg', 15)], settings=[10, 50, 50, 14, 20, 64])

    puzzle_solver = PuzzleSolver("butterfly", (38, 54), 10000, 10000,
                                 [('input/butterfly2_01.jpg', 60), ('input/butterfly2_02.jpg', 77), ('input/butterfly2_03.jpg', 60), ('input/butterfly2_04.jpg', 66),
                                  ('input/butterfly2_05.jpg', 77), ('input/butterfly2_06.jpg', 60), ('input/butterfly2_07.jpg', 70), ('input/butterfly2_08.jpg', 43), ],
                                 settings=[20, 40, 50, 12, 20, 64])

    # puzzle_solver = PuzzleSolver("animals1", (38, 54), 10000, 10000,
    #     [('input/animals_01.jpg', 77), ('input/animals_02.jpg', 77), ('input/animals_03.jpg', 77), ('input/animals_04.jpg', 77),
    #     ('input/animals_05.jpg', 77), ('input/animals_06.jpg', 77), ('input/animals_07.jpg', 47), ('input/animals_08.jpg', 4)],
    #     settings=[20, 40, 40, 12, 20, 64])

    puzzle_solver.solvePuzzle()

# settings: hue, saturation, and value ranges (from bg color) to mask on. Range of number of pixels from side to choose colors on.
# last setting is points per side to detect


class PuzzleSolver:
    def __init__(self, puzzle_name, dims, num_gens, gen_size, image_infos, settings=[10, 50, 50, 12, 20, 32], sides_first=False):
        self.puzzle_name = puzzle_name
        self.num_gens = num_gens
        self.gen_size = gen_size
        self.total_time = 0
        self.sides_first = sides_first
        self.collection = PieceCollection(settings)
        for filename, num_pieces in image_infos:
            self.collection.addPieces(filename, num_pieces)

        self.generation_counter = 0

        # mutation parameters:
        self.min_exp = 0
        self.max_exp = 1  # self.collection.num_pieces_total // 50
        self.exp_function = (lambda x: 1)  # ranges from 0 to 1

        self.edge_count_dict = {}  # used in calculating similarity score

        self.solutions = None
        self.best_solution = None

        self.puzzle_dims = getPuzzleDims(
            dims, self.collection.num_pieces_total)
        # weight dist - how much should the shape of the piece affect the comparison score
        # weight color - how much should the average color along the edge affect the score
        # weight color hist - how much should the distribution of colors along the edge affect the score?
        #                   different from weight color bc this uses a color histogram and has more info
        # weight length diff - how much should the difference in lengths between corners of the edge affect the score
        self.dist_dict, self.sorted_dists, self.buddy_edges, self.empty_edge_dist, self.cutoff = getDistDict(self.collection.pieces,
                                                                                                             weight_dist=300, weight_color=200, weight_color_hist=100, weight_length_diff=300)

        self.doGen0(self.sides_first, set())

    def solvePuzzle(self):
        if self.sides_first:
            while(self.generation_counter < self.num_gens):
                self.doGeneration()
            self.min_exp = 0.2
            self.max_exp = self.collection.num_pieces_total // 25
            self.doGen0(False, self.best_solution.edges)
            self.generation_counter = 1
        while(self.generation_counter < self.num_gens):
            self.doGeneration()
        print(
            f'best score found in {self.num_gens} generations: {self.best_solution.score}')
        print(f'total time: {self.total_time}')
        solution_image = self.best_solution.getSolutionImage()

        h, w, _ = solution_image.shape
        cv2.destroyAllWindows()
        cv2.imshow('best solution', cv2.resize(solution_image,
                   (int(500 * (w / h)), 500), interpolation=cv2.INTER_AREA))
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.imwrite(f'{self.puzzle_name}Solution.jpg', solution_image)

    def solvePuzzle_gui_mode(self):
        if self.sides_first:
            while(self.generation_counter < self.num_gens):
                self.doGeneration()
            self.min_exp = 0.2
            self.max_exp = self.collection.num_pieces_total // 25
            self.doGen0(False, self.best_solution.edges)
            self.generation_counter = 1
        while(self.generation_counter < self.num_gens):
            self.doGeneration()
        print(
            f'best score found in {self.num_gens} generations: {self.best_solution.score}')
        print(f'total time: {self.total_time}')
        solution_image = self.best_solution.getSolutionImage()

        h, w, _ = solution_image.shape
        cv2.destroyAllWindows()
        # cv2.imshow('best solution', cv2.resize(solution_image,
        #            (int(500 * (w / h)), 500), interpolation=cv2.INTER_AREA))
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        cv2.imwrite(f'{self.puzzle_name}Solution.jpg', solution_image)

    def doGeneration(self):
        collection = copy.copy(self.collection)
        if self.sides_first:
            collection.pieces = [
                piece for piece in collection.pieces if piece.type == 'side' or piece.type == 'corner']

        gen_time = 0

        self.solutions = sorted(self.solutions, key=lambda x: x.score)

        include = self.solutions[:self.gen_size//20]

        selection = select_all(
            self.solutions, self.gen_size//2, 2, include=include)
        self.solutions = None
        gc.collect()

        # selection += top_10_percent
        # selection.append(self.best_solution.mutate(0))

        # new_solutions += selection
        # maybe keep top 10% of scores, select remaining 1/2, crossover to fill
        new_solutions = include

        counter = 0
        limit = self.gen_size * 10

        #new_solutions += top_percent
        while len(new_solutions) < self.gen_size and counter < limit:
            start = timer()
            parent1, parent2 = random.choices(selection, k=2)
            if abs(parent1.score - parent2.score) < 1:
                counter += 1
                continue
            solver = parent1.crossover(parent2)

            new_solutions.append(solver)
            #print(f'gen {gen} solution {len(new_solutions)}')
            #print(f'p1 score: {parent1.score}, p2 score: {parent2.score}, c score: {solver.score}')
            end = timer()
            #print(f'time: {end - start}')
            gen_time += (end - start)

            # print('\n')
            if len(new_solutions) % 50 == 0:
                print(f'\t{len(new_solutions)}')

        start = timer()

        self.updateSimilarityScore(new_solutions)
        self.solutions = [solution.mutate(self.getMutationRate(
            solution)) for solution in new_solutions]
        # solutions = new_solutions
        new_scores = [solution.score for solution in self.solutions]

        if min(new_scores) < self.best_solution.score:
            self.best_solution = min(
                self.solutions, key=lambda x: x.score).copy()

            solution_image = self.best_solution.getSolutionImage()
            h, w, _ = solution_image.shape
            #cv2.imshow(f'best solution gen {gen}', cv2.resize(solution_image, (int(500 * (w / h)), 500), interpolation=cv2.INTER_AREA))
            # cv2.waitKey()
            cv2.imwrite(
                f'best_solution_gen_{self.generation_counter}_{self.puzzle_name}.jpg', solution_image)

        end = timer()
        gen_time += (end - start)
        print(f'total time gen {self.generation_counter}: {gen_time}')

        self.solutions = sorted(self.solutions, key=lambda x: x.score)
        scores = [int(solution.score) for solution in self.solutions]
        scores2 = sorted(list(set(scores)))
        diversities = sorted(
            [solution.similarity_score for solution in self.solutions])

        self.updateSimilarityScore(self.solutions)

        print(
            f'solution len edges: {[len(solution.edges) for solution in self.solutions[:10]]}')

        #top_percent = solutions[:len(solutions)//10]
        print(
            f'best score gen {self.generation_counter} : {self.best_solution.score}')
        print(
            f'top 10 gen {self.generation_counter} scores: {scores[:10]},\n top 10 unique: {scores2[:10]}')
        print(
            f'top 10 scores diversities: {[solution.similarity_score for solution in self.solutions[:10]]}')
        print(
            f'top 10 scores mutation rates: {[self.getMutationRate(solution) for solution in self.solutions[:10]]}')
        print(
            f'top 10 scores expected # mutations: {[1 / (1 - self.getMutationRate(solution)) - 1 for solution in self.solutions[:10]]}')
        print(
            f'average score: {sum(scores) // len(scores)}, num varieties: {len(scores2)}\n\n')

        self.generation_counter += 1
        self.total_time += gen_time

        if counter >= limit:
            self.generation_counter = self.num_gens

    def doGen0(self, just_sides, include_edges):
        collection = copy.copy(self.collection)
        self.solutions = []
        side_pieces = [piece for piece in collection.pieces if piece.type ==
                       'side' or piece.type == 'corner']
        if just_sides:
            collection.pieces = [
                piece for piece in collection.pieces if piece.type == 'side' or piece.type == 'corner']
            self.min_exp = 0
            self.max_exp = len(collection.pieces) // 25
        print('beggining genetic process')
        gen_time = 0
        num_mutations = collection.num_pieces_total // 25
        for i in range(self.gen_size):
            start = timer()
            solver = PuzzleSolution(collection, self.puzzle_dims, self.dist_dict,
                                    self.sorted_dists, self.buddy_edges, self.empty_edge_dist, self.cutoff)
            if len(include_edges) > 0:
                side_edges = copy.copy(include_edges)
                for j in range(2):
                    side_edges.remove(random.choice(list(side_edges)))
                solver.solvePuzzle(start=random.choice(
                    side_pieces), include_edges=side_edges)
            else:
                solver.solvePuzzle(random_start=True)
            # image = solver.getSolutionImage()
            # h, w, _ = image.shape
            # cv2.imshow('solution', cv2.resize(image, (int(500 * (w/h)), 500), interpolation=cv2.INTER_AREA))
            # cv2.waitKey(0)

            # solver = solver.mutate(1 - 1 / num_mutations)

            self.solutions.append(solver)
            end = timer()
            gen_time += (end - start)

        self.best_solution = min(self.solutions, key=lambda x: x.score)
        self.updateSimilarityScore(self.solutions)

        solution_image = self.best_solution.getSolutionImage()
        # cv2.imwrite(
        #     f'best_solution_gen_0{self.puzzle_name}.jpg', solution_image)
        print(f'total time gen 0: {gen_time}')

        scores = [int(solution.score) for solution in self.solutions]
        scores = sorted(scores)
        scores2 = sorted(list(set(scores)))

        print(
            f'top 10 gen {0} scores: {scores[:10]},\n top 10 unique: {scores2[:10]}')
        print(
            f'average score: {sum(scores) // len(scores)}, num varieties: {len(scores2)}\n\n')

        self.total_time += gen_time
        self.generation_counter += 1

    def getMutationRate(self, solver):
        # the expected number of mutations will increase
        # btw min and max as the diversity score increases
        # from 0 to 1
        # exp_fun defines the shape of the expected number of mutations curve
        # where the number of mutation rates is how many mutations will occur for a single
        # solution during mutation
        min_exp = self.min_exp
        max_exp = self.max_exp
        # (lambda x:x**3) #exp_fun should be a function that ranges btw 0 and 1
        exp_fun = self.exp_function

        # using some stats stuff. The expected number of mutations follows a
        # geometric distribution, with parameter (1 - p), where p is the
        # probability of a mutation occurring
        # by transforming accordingly, we can ensure that the expected number of
        # mutations follows the function exp_fun, offset to ensure
        # min_exp mutations at d == 0, and max_exp mutations at d == 1
        d = solver.similarity_score
        exp_num_muts = min_exp + exp_fun(d)*(max_exp - min_exp)
        p = 1 - (1 / (1 + exp_num_muts))
        return p

    def updateSimilarityScore(self, solvers):
        self.edge_count_dict = {}
        for solver in solvers:
            for edge in solver.all_edges:
                count = self.edge_count_dict.get(edge)
                if count is None:
                    count = 0
                self.edge_count_dict[edge] = count + 1

        for solver in solvers:
            solver.similarity_score = 0
            for edge in solver.all_edges:
                num_edges = self.edge_count_dict.get(edge)
                solver.similarity_score += (num_edges - 1)
            solver.similarity_score /= (self.gen_size * len(solver.all_edges))
        # if an edge was in every single solution in every generation, then it would add 1 to the diversity score
        # if an edge was in NO others then it would add 1/(gen_size*(gen_counter + 1)) to the total

        # "the average edge in (solver) is in (100 * similarity_score) % of the population seen so far"

        # so the diversity score ranges from 1 / num_edges (if all unique) to 1 if all in every other solution


def select_all(solvers, n, k, include=[]):
    selection = set(include)
    unique_solvers = set()
    scores = set()
    for solver in solvers:
        if not solver.score in scores:
            unique_solvers.add(solver)
            scores.add(solver.score)
    counter = 0
    limit = 10*len(solvers)
    while len(selection) < min(n, len(unique_solvers)):
        selection.add(select_one_by_score(solvers, k))
    return list(selection)


def select_one_by_score(solvers, k):
    random_solvers = random.choices(solvers, k=k)
    best = min(random_solvers, key=lambda x: x.score *
               (1 + x.similarity_score**2))
    return best

# temporary filler function


def getPuzzleDims(size_inches, num_pieces):
    diff = 1000
    side_Ratio = size_inches[1] / size_inches[0]
    predict_Ratio = 0
    predict_w, predict_h = 0, 0

    for i in range(1, int(pow(num_pieces, 1 / 2)) + 1):
        if num_pieces % i == 0:
            if size_inches[1] < size_inches[0]:
                predict_Ratio = i / int(num_pieces / i)
            else:
                predict_Ratio = int(num_pieces / i) / i
            if abs(side_Ratio - predict_Ratio) < diff:
                diff = abs(side_Ratio - predict_Ratio)
                if size_inches[1] < size_inches[0]:
                    predict_w = i
                    predict_h = int(num_pieces / i)
                else:
                    predict_w = int(num_pieces / i)
                    predict_h = i
    # for now just putting the actual number of pieces in as size_inches oops
    return predict_h, predict_w


# returns a dict containing the distances between all edges
def getDistDict(pieces, weight_dist=100, weight_color=100, weight_color_hist=100, weight_length_diff=100):

    dist_dict = {}

    #sorted_dists = {}
    #max_dist = 0
    max_dist_diff = 0
    min_dist_diff = float('inf')
    max_color_diff = 0
    min_color_diff = float('inf')
    max_color_diff_hist = 0
    min_color_diff_hist = float('inf')
    max_corner_diff = 0
    min_corner_diff = float('inf')
    for i, piece1 in enumerate(pieces):
        print(piece1.label)
        for piece2 in pieces[i+1:]:
            for edge1 in range(4):
                for edge2 in range(4):
                    dist = piece1.edges[edge1].compare(piece2.edges[edge2])
                    if dist is None:
                        continue
                    dist_diff, color_diff, color_diff_hist, corner_diff = dist
                    if dist_diff > max_dist_diff:
                        max_dist_diff = dist_diff
                    if dist_diff < min_dist_diff:
                        min_dist_diff = dist_diff
                    if color_diff > max_color_diff:
                        max_color_diff = color_diff
                    if color_diff < min_color_diff:
                        min_color_diff = color_diff
                    if color_diff_hist > max_color_diff_hist:
                        max_color_diff_hist = color_diff_hist
                    if color_diff_hist < min_color_diff_hist:
                        min_color_diff_hist = color_diff_hist
                    if corner_diff > max_corner_diff:
                        max_corner_diff = corner_diff
                    if corner_diff < min_corner_diff:
                        min_corner_diff = corner_diff
                    # if dist < float('inf'):
                        # if dist > max_dist:
                        #max_dist = dist
                        #dist_list.append(((piece2, edge2), dist))
            #sorted_dists[(piece1, edge1)] = sorted(dist_list, key=lambda x:x[1])
    print(max_dist_diff, min_dist_diff, max_color_diff, min_color_diff,
          max_color_diff_hist, min_color_diff_hist, max_corner_diff, min_corner_diff)

    num_middle_pieces = len(
        [piece for piece in pieces if piece.type == 'middle'])
    num_side_pieces = len([piece for piece in pieces if piece.type == 'side'])

    num_edges = 4*num_middle_pieces + 3*num_side_pieces + 2*4

    best_sides = [float('inf') for _ in range(num_edges*2)]

    sorted_dists = {}

    for piece1 in pieces:
        for edge1 in range(4):
            sorted_dists[(piece1, edge1)] = [((None, None), float('inf'))
                                             for _ in range(len(pieces) // 4)]

    max_dist = 0
    for i, piece1 in enumerate(pieces):
        print(piece1.label)
        for edge1 in range(4):
            dist_list = []
            for piece2 in pieces[i+1:]:
                for edge2 in range(4):
                    entry = piece1.edges[edge1].compare(piece2.edges[edge2])
                    if entry is None:
                        dist_dict[(piece1, edge1, piece2, edge2)
                                  ] = float('inf')
                        continue
                    dist_diff, color_diff, color_diff_hist, corner_diff = entry
                    dist_diff = (dist_diff - min_dist_diff) / \
                        (max_dist_diff - min_dist_diff)
                    color_diff = (color_diff - min_color_diff) / \
                        (max_color_diff - min_color_diff)
                    color_diff_hist = (color_diff_hist - min_color_diff_hist) / \
                        (max_color_diff_hist - min_color_diff_hist)
                    corner_diff = (corner_diff - min_corner_diff) / \
                        (max_corner_diff - min_corner_diff)

                    dist = weight_dist*dist_diff + weight_color*color_diff + \
                        weight_color_hist*color_diff_hist + weight_length_diff*corner_diff
                    dist_dict[(piece1, edge1, piece2, edge2)] = dist
                    if dist < float('inf'):
                        if dist > max_dist:
                            max_dist = dist
                        if dist < best_sides[-1]:
                            best_sides[-1] = dist
                            best_sides = sorted(best_sides)
                        if dist < sorted_dists[(piece1, edge1)][-1][1]:
                            sorted_dists[(piece1, edge1)
                                         ][-1] = ((piece2, edge2), dist)
                            sorted_dists[(piece1, edge1)] = sorted(
                                sorted_dists[(piece1, edge1)], key=lambda x: x[1])
                        if dist < sorted_dists[(piece2, edge2)][-1][1]:
                            sorted_dists[(piece2, edge2)
                                         ][-1] = ((piece1, edge1), dist)
                            sorted_dists[(piece2, edge2)] = sorted(
                                sorted_dists[(piece2, edge2)], key=lambda x: x[1])
    cutoff = best_sides[-1] + 1

    for piece1 in pieces:
        for edge1 in range(4):
            sorted_dists[(piece1, edge1)] = [
                entry for entry in sorted_dists[(piece1, edge1)] if entry[1] != float('inf')]

    buddy_edges = set()
    for piece1, edge1 in sorted_dists.keys():
        piece1.edges[edge1].clear()
        sorted_dists[(piece1, edge1)] = sorted(
            sorted_dists[(piece1, edge1)], key=lambda x: x[1])
        entries = sorted_dists[(piece1, edge1)][:3]
        for entry, dist in entries:
            # entry, dist = sorted_dists[(piece1, edge1)]
            piece2, edge2 = entry
            sorted_dists[(piece2, edge2)] = sorted(
                sorted_dists[(piece2, edge2)], key=lambda x: x[1])
            if (piece1, edge1) == sorted_dists[(piece2, edge2)][0][0]:
                buddy_edges.add((piece1, edge1, piece2, edge2))
    return dist_dict, sorted_dists, buddy_edges, max_dist, cutoff


if __name__ == "__main__":
    main()
