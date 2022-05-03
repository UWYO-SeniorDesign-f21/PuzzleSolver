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
from compressed_dictionary import CompressedDictionary as cdict

def main():
    '''
    Uncomment the lines for the puzzle you want to run! Refer to the constructor for PuzzleSolver to
    change parameters / settings etc
    '''

    # puzzle_solver = PuzzleSolver("bluesclues", (4, 6), 30, 30,
    #                             [('input/puzzle1_1.jpg', 24)], settings=[10, 50, 50, 10, 20, 64])

    # puzzle_solver = PuzzleSolver("superman", (6, 8), 30, 30,
    #                             [('input/puzzle2_1.jpg', 48)], settings=[10, 50, 50, 10, 20, 64])

    # puzzle_solver = PuzzleSolver("starwars", (6, 8), 30, 30,
    #                              [('input/star_wars_new.jpg', 48)], settings=[10, 50, 50, 5, 10, 64])

    # puzzle_solver = PuzzleSolver("pokemon_1", (14.7, 10.5), 30, 30,
    #     [('input/pokemon_puzzle_1_01.png', 20),
    #     ('input/pokemon_puzzle_1_02.png', 20),
    #     ('input/pokemon_puzzle_1_03.png', 20),
    #     ('input/pokemon_puzzle_1_04.png', 20),
    #     ('input/pokemon_puzzle_1_05.png', 20)], settings=[10, 50, 50, 14, 20, 32], sides_first=False)

    # puzzle_solver = PuzzleSolver("pokemon_2", (15, 11), 100, 100,
    #     [('input/pokemon_puzzle_2_01.png', 20),
    #     ('input/pokemon_puzzle_2_02.png', 20),
    #     ('input/pokemon_puzzle_2_03.png', 20),
    #     ('input/pokemon_puzzle_2_04.png', 20),
    #     ('input/pokemon_puzzle_2_05.png', 20)], settings=[10, 50, 50, 14, 20, 32])


    # puzzle_solver = PuzzleSolver("beachHut", (21.25, 15), 100, 100,
    #     [('input/300_01.png', 30), ('input/300_02.png', 30), ('input/300_03.png', 30), ('input/300_04.png', 30),
    #     ('input/300_05.png', 30), ('input/300_06.png', 30), ('input/300_07.png', 30), ('input/300_08.png', 30),
    #     ('input/300_09.png', 30), ('input/300_10.png', 30)], settings=[10, 25, 30, 8, 14, 32])

    # puzzle_solver = PuzzleSolver("tart7", (18, 18), 50, 500, [('input/tart_puzzle_01.jpg', 30), ('input/tart_puzzle_02.jpg', 30), ('input/tart_puzzle_03.jpg', 30), ('input/tart_puzzle_04.jpg', 30), ('input/tart_puzzle_05.jpg', 30), (
    #     'input/tart_puzzle_06.jpg', 30), ('input/tart_puzzle_07.jpg', 28), ('input/tart_puzzle_08.jpg', 30), ('input/tart_puzzle_09.jpg', 30), ('input/tart_puzzle_10.jpg', 30), ('input/tart_puzzle_11.jpg', 26)], settings=[10, 50, 50, 8, 14, 64],
    #     sides_first=False)

    # puzzle_solver = PuzzleSolver("feather", (21.25, 15), 200, 500,
    #     [('input/feather2_01.jpg', 40),('input/feather2_02.jpg', 40),('input/feather2_03.jpg', 40),('input/feather2_04.jpg', 40),
    #     ('input/feather2_05.jpg', 40),('input/feather2_06.jpg', 40),('input/feather2_07.jpg', 40),('input/feather2_08.jpg', 20)],
    #     settings=[20, 40, 50, 12, 16, 32], sides_first=False)

    # puzzle_solver = PuzzleSolver("travel", (21.25, 15), 200, 500,
    #     [('input/travel_puzzle_01.jpg', 30),('input/travel_puzzle_02.jpg', 30),('input/travel_puzzle_03.jpg', 30),('input/travel_puzzle_04.jpg', 30),
    #     ('input/travel_puzzle_05.jpg', 30),('input/travel_puzzle_06.jpg', 30),('input/travel_puzzle_07.jpg', 30),('input/travel_puzzle_08.jpg', 30),
    #     ('input/travel_puzzle_09.jpg', 30),('input/travel_puzzle_10.jpg', 12),('input/travel_puzzle_11.jpg', 18)],
    #      settings=[10, 50, 50, 10, 20, 64], sides_first=False)

    # puzzle_solver = PuzzleSolver("owl3", (21.25, 15), 200, 500,
    # [('input/owl2_01.jpg', 40), ('input/owl2_02.jpg', 40), ('input/owl2_03.jpg', 40),
    # ('input/owl2_04.jpg', 40), ('input/owl2_05.jpg', 40), ('input/owl2_06.jpg', 40),
    # ('input/owl2_07.jpg', 40), ('input/owl2_08.jpg', 20)], settings=[10, 30, 30, 20, 30, 64], sides_first=False)

    # puzzle_solver = PuzzleSolver("shining", (18, 24), 10000, 10000,
    #     [('input/shining_01.jpg', 42), ('input/shining_02.jpg', 42), ('input/shining_03.jpg', 42), ('input/shining_04.jpg', 42),
    #     ('input/shining_05.jpg', 42), ('input/shining_06.jpg', 42), ('input/shining_07.jpg', 42), ('input/shining_08.jpg', 42),
    #     ('input/shining_09.jpg', 42), ('input/shining_10.jpg', 42), ('input/shining_11.jpg', 42), ('input/shining_12.jpg', 23),
    #     ('input/shining_13.jpg', 15)], settings=[10, 50, 50, 14, 20, 64])

    # puzzle_solver = PuzzleSolver("butterfly", (38, 54), 10000, 10000,
    #     [('input/butterfly2_01.jpg', 60), ('input/butterfly2_02.jpg', 77), ('input/butterfly2_03.jpg', 60), ('input/butterfly2_04.jpg', 66),
    #     ('input/butterfly2_05.jpg', 77), ('input/butterfly2_06.jpg', 60), ('input/butterfly2_07.jpg', 70), ('input/butterfly2_08.jpg', 43), ],
    #     settings=[20, 40, 50, 16, 26, 64])

    # puzzle_solver = PuzzleSolver("animals1", (38, 54), 10000, 10000,
    #     [('input/animals_01.jpg', 77), ('input/animals_02.jpg', 77), ('input/animals_03.jpg', 77), ('input/animals_04.jpg', 77),
    #     ('input/animals_05.jpg', 77), ('input/animals_06.jpg', 77), ('input/animals_07.jpg', 47), ('input/animals_08.jpg', 4)],
    #     settings=[20, 40, 40, 12, 20, 64])

    # puzzle_solver = PuzzleSolver("pokemonBeach", (54, 38), 200, 1000, 
    #     [('input/pokemonBeach01.png', 48), ('input/pokemonBeach02.png', 48), ('input/pokemonBeach03.png', 48),
    #     ('input/pokemonBeach04.png', 48), ('input/pokemonBeach05.png', 48), ('input/pokemonBeach06.png', 48),
    #     ('input/pokemonBeach07.png', 42), ('input/pokemonBeach08.png', 48), ('input/pokemonBeach09.png', 48),
    #     ('input/pokemonBeach10.png', 34), ('input/pokemonBeach11.png', 34), ('input/pokemonBeach12.png', 19)],
    #     settings=[10, 25, 30, 6, 20, 64])

    # puzzle_solver = PuzzleSolver("waterfront11", (54, 38), 300, 1500, 
    #     [('input/waterfront01.png', 48), ('input/waterfront02.png', 48), ('input/waterfront03.png', 48),
    #     ('input/waterfront04.png', 48), ('input/waterfront05.png', 42), ('input/waterfront06.png', 42),
    #     ('input/waterfront07.png', 48), ('input/waterfront08.png', 48), ('input/waterfront09.png', 48),
    #     ('input/waterfront10.png', 33), ('input/waterfront11.png', 12), ('input/waterfront12.png', 48)],
    #     settings=[30, 50, 50, 8, 14, 64], sides_first=False)

    # puzzle_solver = PuzzleSolver("donut", (38, 27), 500, 1500,
    #     [('input/donut01.png', 48), ('input/donut02.png', 48), ('input/donut03.png', 48), ('input/donut04.png', 48),
    #     ('input/donut05.png', 48), ('input/donut06.png', 48), ('input/donut07.png', 48), ('input/donut08.png', 41),
    #     ('input/donut09.png', 48), ('input/donut10.png', 48), ('input/donut11.png', 47), ('input/donut12.png', 48),
    #     ('input/donut13.png', 48), ('input/donut14.png', 48), ('input/donut15.png', 48), ('input/donut16.png', 48),
    #     ('input/donut17.png', 48), ('input/donut18.png', 48), ('input/donut19.png', 48), ('input/donut20.png', 9),
    #     ('input/donut21.png', 48), ('input/donut22.png', 29), ('input/donut23.png', 36)], 
    #     settings=[30, 50, 50, 10, 16, 64], sides_first=False)

    # puzzle_solver = PuzzleSolver("patches", (56, 37), 1000, 200,
    #     [('input/patches01.png', 42), ('input/patches02.png', 48), ('input/patches03.png', 47), ('input/patches04.png', 48),
    #     ('input/patches05.png', 48), ('input/patches06.png', 48), ('input/patches07.png', 48), ('input/patches08.png', 47),
    #     ('input/patches09.png', 48), ('input/patches10.png', 48), ('input/patches11.png', 48), ('input/patches12.png', 48),
    #     ('input/patches13.png', 48), ('input/patches14.png', 48), ('input/patches15.png', 48), ('input/patches16.png', 14),
    #     ('input/patches17.png', 48), ('input/patches18.png', 48), ('input/patches19.png', 48), ('input/patches20.png', 48),
    #     ('input/patches21.png', 48), ('input/patches22.png', 47), ('input/patches23.png', 48), ('input/patches24.png', 48), 
    #     ('input/patches25.png', 48), ('input/patches26.png', 48), ('input/patches27.png', 47), ('input/patches28.png', 48),
    #     ('input/patches29.png', 48), ('input/patches30.png', 48), ('input/patches31.png', 48), ('input/patches32.png', 48), 
    #     ('input/patches33.png', 48), ('input/patches34.png', 47), ('input/patches35.png', 48), ('input/patches36.png', 48), 
    #     ('input/patches37.png', 48), ('input/patches38.png', 48), ('input/patches39.png', 47), ('input/patches40.png', 48), 
    #     ('input/patches41.png', 48), ('input/patches42.png', 48), ('input/patches43.png', 48), ('input/patches44.png', 48),
    #     ('input/patches45.png', 6)],
    #     settings=[30, 50, 50, 10, 20, 64], sides_first=True)

    # # # # # use LAB AND increase color weight, decrease dist weight BC the corners on this one DONT line up

    # puzzle_solver = PuzzleSolver("market10", (15, 21), 500, 500,
    #     [('input/market01.png', 48), ('input/market02.png', 48), ('input/market03.png', 48), ('input/market04.png', 48),
    #     ('input/market05.png', 37), ('input/market06.png', 36), ('input/market07.png', 48), ('input/market08.png', 48), 
    #     ('input/market09.png', 48), ('input/market10.png', 43), ('input/market11.png', 48)],
    #     settings=[30, 30, 30, 6, 20, 64], color_spec="BGR", sides_first=True)

# settings: hue, saturation, and value ranges (from bg color) to mask on. Range of number of pixels from side to choose colors on.
# last setting is points per side to detect

    puzzle_solver.solvePuzzle()
'''
The PuzzleSolver class controls the genetic algorithm and the solving process. 
Passed are the puzzle name, dimensions (in inches), parameters for the genetic algorithm
and settings for finding the pieces / info about the pieces

Settings are:
1-3: range of color values to mask on in each of the three color channels
4-5: range of pixels in from the contour to find colors over

has an option to do sides first, where the sides of the puzzle are solved and then
used in future solutions
'''
class PuzzleSolver:
    def __init__(self, puzzle_name, dims, num_gens, gen_size, image_infos, show_sols=True, settings=[10, 50, 50, 20, 30, 64], color_spec="HSV", sides_first=False):
        self.side_gen_size = 500 # if doing sides first
        self.side_gens = 500
        
        self.puzzle_name = puzzle_name
        self.num_gens = num_gens
        self.gen_size = gen_size
        self.total_time = 0
        self.sides_first = sides_first

        self.collection = PieceCollection(settings)
        # add pieces to collection
        for filename, num_pieces in image_infos:
            self.collection.addPieces(filename, num_pieces, color_spec=color_spec)
        
        # number of generations done so far
        self.generation_counter = 0

        # start time of the solver
        self.begin_timer = timer()

        # mutation parameters:
        self.min_exp = 0
        self.max_exp = max(1, len(self.collection.pieces) // 50)
        self.exp_function = (lambda x: 1)#max( 0.9 * (x > 0.9), (self.num_gens - self.generation_counter) / (self.num_gens)))  # ranges from 0 to 1
        self.move_up_factor = (lambda x: 0.2)# 0.3 * ((self.num_gens - x) / (self.num_gens)))

        self.selection_size = min(self.gen_size//2, 100) # how many solutions to select from each generation


        self.edge_count_dict = {}  # used in calculating similarity score

        self.solutions = []
        self.best_solution = None

        self.show_sols = show_sols

        # calculate puzzle dimensions, in number of pieces per side
        self.puzzle_dims = getPuzzleDims(
            dims, self.collection.num_pieces_total)
 
        # if doing sides first
        if self.sides_first:
            self.solvePuzzleSides()
            self.solved_sides = self.best_solution.edges.copy()
            self.solutions = []
            self.best_solution = None
        
        # fix piece numbers from sides first / store numbers for pieces
        for i, piece in enumerate(self.collection.pieces):
            piece.number = i

        # weight dist - how much should the shape of the piece affect the comparison score
        # weight color - how much should the average color along the edge affect the score
        # weight color hist - how much should the distribution of colors along the edge affect the score?
        #                   different from weight color bc this uses a color histogram and has more info
        # weight length diff - how much should the difference in lengths between corners of the edge affect the score
        # num_edges_to_include - how many edges to save in sorted order, greater = more memory use
        # save_all_dists - turn to False to calculate non-sorted dists as the solver goes. False = slow but low memory usage


        self.dist_dict, self.sorted_dists, self.buddy_edges, self.empty_edge_dist, self.cutoff = getDistDict(self.collection.pieces,
                weight_dist=3, weight_color=2, weight_color_hist=1, weight_length_diff=3, num_edges_to_include=200, store_all_dists=True)

        gc.collect()

    '''
    Solves the puzzle with just the sides. Gives a lot less information so can be
    hard for it to solve the sides, but if it gets them close can help a lot with the
    rest of the process.
    '''
    def solvePuzzleSides(self):
        self.side_collection = copy.copy(self.collection)
        self.side_collection.pieces = [piece for piece in self.collection.pieces if piece.type == 'side' or piece.type == 'corner']
        for i, piece in enumerate(self.side_collection.pieces):
            piece.number = i
        self.dist_dict, self.sorted_dists, self.buddy_edges, self.empty_edge_dist, self.cutoff = \
            getDistDict(self.side_collection.pieces, \
            weight_dist=3, weight_color=2, weight_color_hist=1, weight_length_diff=3, num_edges_to_include=50, store_all_dists=True)
        
        prev_max_exp = self.max_exp
        prev_gen_size = self.gen_size
        self.gen_size = self.side_gen_size
        self.max_exp = len(self.side_collection.pieces) // 10
        self.puzzle_name = f'{self.puzzle_name}Sides'
        self.doGen0(self.sides_first, set())
        prev_best = round(self.best_solution.score, 3)

        while self.generation_counter < self.side_gens:
            self.doGeneration()

            print(f'\nTime since started: {timer() - self.begin_timer:.2f}')
        
        for i, piece in enumerate(self.collection.pieces):
            piece.number = i

        self.generation_counter = 0
        self.gen_size = prev_gen_size
        gc.collect()

    '''
    Solve the puzzle regular mode
    '''
    def solvePuzzle(self):
        if self.sides_first:
            self.doGen0(False, self.solved_sides)
            self.sides_first = False
        else:
            self.doGen0(False, set())

        while True:
            self.doGeneration()

            print(f'\nTime since started: {timer() - self.begin_timer:.2f}\n')

            diversities = [solution.similarity_score for solution in self.solutions]

            # if all the solutions are basically the same, then it is done
            if sum(diversities) / len(diversities) >= (self.collection.num_pieces_total - 1)/(self.collection.num_pieces_total):
                break

        print(
            f'best score found in {self.num_gens} generations: {self.best_solution.score}')
        print(f'total time: {self.total_time}')

        try:
            solution_image = self.best_solution.getSolutionImage(resize_factor=1, just_sides=self.sides_first)
            h, w, _ = solution_image.shape

            cv2.imwrite('result.jpg', solution_image)

            cv2.imshow('best solution', cv2.resize(solution_image,
                    (int(500 * (w / h)), 500), interpolation=cv2.INTER_AREA))
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.imwrite(f'{self.puzzle_name}Solution.jpg', solution_image)

        except Exception as e: 
            print(e)

    '''
    Used to interface with the GUI
    '''
    def solvePuzzle_gui_mode(self):
        if self.sides_first:
            self.doGen0(False, self.solved_sides)
            self.generation_counter = 1
        else:
            self.doGen0(False, set())

        while(self.generation_counter < self.num_gens):
            self.doGeneration()
            solution_image = self.best_solution.getSolutionImage(resize_factor=1, just_sides=self.sides_first)
            try:
                solution_image = self.best_solution.getSolutionImage(resize_factor=1, just_sides=self.sides_first)
                h, w, _ = solution_image.shape

                cv2.imwrite('result.jpg', solution_image)
            except Exception as e: 
                print(e)

            print(f'\nTime since started: {timer() - self.begin_timer:.2f}\n')

            diversities = [solution.similarity_score for solution in self.solutions]

            # if all the solutions are basically the same, then it is done
            if sum(diversities) / len(diversities) >= (self.collection.num_pieces_total - 1)/(self.collection.num_pieces_total):
                break

        print(
            f'best score found in {self.num_gens} generations: {self.best_solution.score}')
        print(f'total time: {self.total_time}')
        try:
            solution_image = self.best_solution.getSolutionImage(resize_factor=1, just_sides=self.sides_first)
            h, w, _ = solution_image.shape

            cv2.imwrite('result.jpg', solution_image)
        except Exception as e: 
            print(e)


    '''
    performs one generation of the genetic algorithm
    '''
    def doGeneration(self):
        start = timer()
        gen_time = 0

        self.solutions = sorted(self.solutions, key=lambda x: x.score)

        selection_size = self.selection_size

        include = self.solutions[:max(2, selection_size//20)]
        selection = self.select_all(self.solutions, selection_size, self.gen_size//selection_size, include=include, move_up_factor=self.move_up_factor(self.generation_counter))

        print(sorted([solver.score for solver in selection]))

        self.solutions = None
        gc.collect()

        for solution in selection:
            before = solution.score
            if solution in include:
                max_mutation_score = 0
            else:
                max_mutation_score = self.empty_edge_dist
            solution.mutate(self.getMutationRate(solution), max_mutation_score=max_mutation_score)
  
        self.solutions = include.copy()

        counter = 0
        limit = self.gen_size * 10

        while len(self.solutions) < self.gen_size and counter < limit:
            parent1, parent2 = random.choices(selection, k=2)

            solver = parent1.crossover(parent2, just_sides=self.sides_first)

            self.solutions.append(solver)

            if len(self.solutions) % 50 == 0:
                print(f'\t{len(self.solutions)}')

        self.updateSimilarityScore(self.solutions, include)

        new_scores = sorted([solution.score for solution in self.solutions])

        self.solutions = sorted(self.solutions, key=lambda x: x.score)

        if min(new_scores) < self.best_solution.score:
            self.best_solution = min(
                self.solutions, key=lambda x: x.score).copy()

            try:

                if self.show_sols:
                    solution_image = self.best_solution.getSolutionImage(resize_factor=1, just_sides=self.sides_first)
                    h, w, _ = solution_image.shape

                    cv2.imwrite(
                        f'best_solution_gen_{self.generation_counter}_{self.puzzle_name}.jpg', solution_image)
           
            except Exception as e: 
                print(e)
        
        end = timer()
        gen_time = (end - start)
        print(f'total time gen {self.generation_counter}: {gen_time}')

        scores = [solution.score for solution in self.solutions]
        scores2 = sorted(list(set(scores)))
        diversities = sorted(
            [solution.similarity_score for solution in self.solutions])

        print(
            f'solution len edges: {[len(solution.edges) for solution in self.solutions[:10]]}')
        print(
            f'best score gen {self.generation_counter} : {self.best_solution.score}')
        print(
            f'top 10 gen {self.generation_counter} scores: {[int(score) for score in scores[:10]]}')
        print(
            f'top 10 scores diversities: {[solution.similarity_score for solution in self.solutions[:10]]}')
        print(
            f'top 10 scores mutation rates: {[self.getMutationRate(solution) for solution in self.solutions[:10]]}')
        print(
            f'top 10 scores expected # mutations: {[1 / (1 - self.getMutationRate(solution)) - 1 for solution in self.solutions[:10]]}')
        print(
            f'average score: {sum(scores) // len(scores)}, avg sim: {sum(diversities) / len(diversities) :.2f}, num varieties: {len(scores2)}\n\n')

        self.generation_counter += 1
        self.total_time += gen_time
    
    '''
    Initializes the genetic algorithm
    '''
    def doGen0(self, just_sides, include_edges):
        if just_sides:
            collection = self.side_collection
        else:
            prev_side_indexes = None
            collection = self.collection

        print('beggining genetic process')
        gen_time = 0
        num_mutations = len(collection.pieces) // 16
        for i in range(self.gen_size):
            if i % 50 == 0:
                print(i)
            start = timer()
            if just_sides:
                solver = PuzzleSolution(collection, self.puzzle_dims, self.dist_dict, self.sorted_dists, self.buddy_edges, self.empty_edge_dist, self.cutoff)
            else:
                solver = PuzzleSolution(collection, self.puzzle_dims, self.dist_dict, self.sorted_dists, self.buddy_edges, self.empty_edge_dist, self.cutoff)
            if len(include_edges) > 0:
                choices = set(edge[0] for edge in include_edges)
                new_include_edges = include_edges.copy()
                for _ in range(len(include_edges) // 10):
                    new_include_edges.remove(random.choice(list(new_include_edges)))
                solver.solvePuzzle(start=random.choice(
                    list(choices)), include_edges=include_edges, just_sides=just_sides)
            else:
                solver.solvePuzzle(random_start=True, just_sides=just_sides)

            self.solutions.append(solver)
            end = timer()
            gen_time += (end - start)

        self.best_solution = min(self.solutions, key=lambda x: x.score).copy()
        self.updateSimilarityScore(self.solutions, [self.best_solution])

        try:

            if self.show_sols:
                solution_image = self.best_solution.getSolutionImage(resize_factor=1, just_sides=self.sides_first)
                h, w, _ = solution_image.shape
                cv2.imwrite(
                    f'best_solution_gen_{self.generation_counter}_{self.puzzle_name}.jpg', solution_image)
        except Exception as e: 
            print(e)

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
        # exp_fun should be a function that ranges btw 0 and 1
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

    def updateSimilarityScore(self, solvers, include):

        self.edge_count_dict = {}
        for solver in include:
            for edge in solver.all_edges:
                count = self.edge_count_dict.get(edge)
                if count is None:
                    count = 0
                self.edge_count_dict[edge] = count + 1

        for solver in solvers:
            solver.similarity_score = 0
            for edge in solver.all_edges:
                num_edges = self.edge_count_dict.get(edge)
                if num_edges is None:
                    continue
                solver.similarity_score += num_edges
            solver.similarity_score /= (len(include) * max(1, len(solver.all_edges)))
        # if an edge was in every single solution in every generation, then it would add 1 to the diversity score
        # if an edge was in NO others then it would add 1/(gen_size*(gen_counter + 1)) to the total

        # "the average edge in (solver) is in (100 * similarity_score) % of the population seen so far"

        # so the diversity score ranges from 1 / num_edges (if all unique) to 1 if all in every other solution

    '''
    Performs selection, based on scores and similarities
    '''
    def select_all(self, solvers, n, k, include, move_up_factor=0.5):
        selection = set(include)
        max_include_score = self.best_solution.score
        max_similarity = max(solvers, key=lambda x:x.similarity_score).similarity_score
        min_similarity = min(solvers, key=lambda x:x.similarity_score).similarity_score

        while len(selection) < n:
            selected = select_one_by_score(solvers, k, max_include_score, move_up_factor, min_similarity, max_similarity)
            if selected in selection:
                continue
            selection.add(selected)
        return list(selection)


def select_one_by_score(solvers, k, min_score, move_up_factor, min_similarity=0, max_similarity=1):
    if max_similarity == min_similarity:
        return random.choice(solvers)
    random_solvers_nums = random.choices(range(len(solvers)), k=k)

    best = min(random_solvers_nums, key=lambda x: (solvers[x].score) - \
            (move_up_factor * (solvers[x].score - min_score)) * \
            max(1, (1 - ((solvers[x].similarity_score - min_similarity) \
            / (max_similarity - min_similarity)))))
    return solvers[best]

'''
Returns the puzzle dimensions in number of pieces per side,
using info from the user of the number of inches per side
'''
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

    return predict_h, predict_w


# returns a dict containing the distances between all edges
def getDistDict(pieces, weight_dist=100, weight_color=100, weight_color_hist=100, weight_length_diff=100, num_edges_to_include=50, store_all_dists=True):

    dist_dict = {}
    max_dist_diff = 0
    min_dist_diff = float('inf')
    max_color_diff = 0
    min_color_diff = float('inf')
    max_color_diff_hist = 0
    min_color_diff_hist = float('inf')
    max_corner_diff = 0
    min_corner_diff = float('inf')
    print('initial dists\n')

    for i, piece1 in enumerate(pieces):
        print(piece1.label, end=' ', flush=True)
        for j, piece2 in enumerate(pieces[i+1:]):
            if piece1 == piece2:
                continue
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

    print(max_dist_diff, min_dist_diff, max_color_diff, min_color_diff,
          max_color_diff_hist, min_color_diff_hist, max_corner_diff, min_corner_diff)

    num_middle_pieces = len(
        [piece for piece in pieces if piece.type == 'middle'])
    num_side_pieces = len([piece for piece in pieces if piece.type == 'side'])

    num_edges = 4*num_middle_pieces + 3*num_side_pieces + 2*4

    sorted_dists = {}
    side_num_edges_to_include = num_edges_to_include


    for piece1 in pieces:
        for edge1 in range(4):
            if piece1.edges[edge1].left_neighbor.label == 'flat' or piece1.edges[edge1].right_neighbor.label == 'flat':
                sorted_dists[hash2(piece1, edge1)] = listOfKElements(side_num_edges_to_include)
            elif piece1.edges[edge1].label != 'flat':
                sorted_dists[hash2(piece1, edge1)] = listOfKElements(num_edges_to_include)

    print('\n\nnormalizing ... \n')
    max_dist = 0
    for i, piece1 in enumerate(pieces):
        print(piece1.label, end=' ', flush=True)
        for edge1 in range(4):
            for j, piece2 in enumerate(pieces[i+1:]):
                for edge2 in range(4):
                    entry = piece1.edges[edge1].compare(piece2.edges[edge2])
                    if entry is None:
                        continue
                    dist_diff, color_diff, color_diff_hist, corner_diff = entry
                    dist_diff = (dist_diff - min_dist_diff) / (max_dist_diff - min_dist_diff)
                    color_diff = (color_diff - min_color_diff) / (max_color_diff - min_color_diff)
                    color_diff_hist = (color_diff_hist - min_color_diff_hist) / (max_color_diff_hist - min_color_diff_hist)
                    corner_diff = (corner_diff - min_corner_diff) / (max_corner_diff - min_corner_diff)

                    dist = weight_dist*dist_diff + weight_color*color_diff + weight_color_hist*color_diff_hist + weight_length_diff*corner_diff
                    if store_all_dists:
                        dist_dict[(piece1, edge1, piece2, edge2)] = dist
                    if dist < float('inf'):
                        if dist > max_dist:
                            max_dist = dist
                        sorted_dists[hash2(piece1, edge1)].add(hash2(piece2, edge2), dist)
                        sorted_dists[hash2(piece2, edge2)].add(hash2(piece1, edge1), dist)
            # gc.collect()
    cutoff = float('inf')

    best_edges = {}
    max_num_edges_to_check = 1

    print('\n\nsorting ...\n')
    for piece1 in pieces:
        print(piece1.label, end=' ', flush=True)
        for edge1 in range(4):
            piece1.edges[edge1].weights = [weight_dist, weight_color, weight_color_hist, weight_length_diff]
            piece1.edges[edge1].mins = [min_dist_diff, min_color_diff, min_color_diff_hist, min_corner_diff]
            piece1.edges[edge1].maxs = [max_dist_diff, max_color_diff, max_color_diff_hist, max_corner_diff]
            entry = sorted_dists.get(hash2(piece1, edge1))
            if entry is None:
                continue
            values = entry.getValues()
            if not store_all_dists:
                dists = entry.getDists()
                for i, dist in enumerate(dists):
                    value = values[i]
                    dist_dict[(piece1, edge1, pieces[value//4], value%4)] = dist
            sorted_dists[hash2(piece1, edge1)] = values
            best_edges[(piece1, edge1)] = sorted_dists[hash2(piece1, edge1)][:max_num_edges_to_check]
    
    print(f'\n{max_dist}\n{len(dist_dict.keys())}\n\n')

    for piece1 in pieces:
        for edge1 in range(4):
            values = sorted_dists.get(hash2(piece1, edge1))
            if values is None:
                continue
            for edge_index in values:
                piece2 = pieces[edge_index // 4]
                edge2 = edge_index % 4
                if sorted_dists.get(hash2(piece2, edge2)) is None:
                    continue
                if not hash2(piece1, edge1) in sorted_dists[hash2(piece2, edge2)]:
                    sorted_dists[hash2(piece2, edge2)].append(hash2(piece1, edge1))

    for piece1 in pieces:
        for edge1 in range(4):
            sorted_list = sorted_dists.get(hash2(piece1, edge1))
            if sorted_list is None:
                continue
            sorted_dists[(piece1, edge1)] = sorted(sorted_list, key=lambda x:getDist(dist_dict, (piece1, edge1, pieces[x//4], x%4)))
    print('\n\nfinding best buddies\n')
    buddy_edges_set = set()
    prev_piece = None
    for piece1, edge1 in best_edges.keys():
        if piece1 != prev_piece:
            print(piece1.label, end=' ', flush=True)
            prev_piece = piece1
        if piece1.edges[edge1].left_neighbor.label == 'flat' or piece1.edges[edge1].right_neighbor.label == 'flat':
            num_edges_to_check = 1
        else:
            num_edges_to_check = max_num_edges_to_check
        entries = best_edges[(piece1, edge1)][:num_edges_to_check]
        p1_index = piece1.number * 4 + edge1
        for edge_index in entries:
            piece2 = pieces[edge_index // 4]
            edge2 = edge_index % 4
            if best_edges[(piece2, edge2)][0] == p1_index:
                buddy_edges_set.add(((piece1, edge1, piece2, edge2), dist))
                break
    buddy_edges_list = sorted([buddy for buddy in buddy_edges_set], key=lambda x:x[1])
    buddy_edges = [buddy[0] for buddy in buddy_edges_list]
    best_edges = None
    gc.collect()
    print(f'\n\n\nnum best buddies::: {len(buddy_edges)}\n\n')
    return dist_dict, sorted_dists, buddy_edges, max_dist, cutoff

def hash2(piece1, edge1):
    return piece1.number * 4 + edge1

def getDist(dist_dict, edge):
    if edge[0] == edge[2]:
        return float('inf')
    res = dist_dict.get(edge) if dist_dict.get(edge) else dist_dict.get((edge[2], edge[3], edge[0], edge[1]))
    if res is None:
        return float('inf')
    return res

class listOfKElements():

    def __init__(self, k):
        self.k = k
        self.value_list = []
        self.dist_list = []
        self.max_dist = float('-inf')
        self.max_dist_pos = 0

    def addNoBlock(self, value, dist):
        if dist > self.max_dist:
            self.max_dist = dist
            self.max_dist_pos = len(self.value_list)
        self.value_list.append(value)
        self.dist_list.append(dist)

    def add(self, value, dist):
        if len(self.value_list) < self.k:
            self.addNoBlock(value, dist)
            return
        if dist < self.max_dist:
            self.max_dist = dist
            self.value_list[self.max_dist_pos] = value
            self.dist_list[self.max_dist_pos] = dist
            new_max = float('-inf')
            new_max_pos = 0
            for i, dist in enumerate(self.dist_list):
                if dist > new_max:
                    new_max = dist
                    new_max_pos = i
            self.max_dist = new_max
            self.max_dist_pos = new_max_pos

    def getValues(self):
        order = sorted(range(len(self.dist_list)), key=lambda x:self.dist_list[x])
        sorted_values = [self.value_list[i] for i in order]
        return sorted_values

    def getDists(self):
        return sorted(self.dist_list)


if __name__ == "__main__":
    main()
