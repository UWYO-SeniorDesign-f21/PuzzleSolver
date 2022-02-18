from puzzleSolution import PuzzleSolution
from pieceCollection import PieceCollection
import random
import cv2
import numpy as np
from timeit import default_timer as timer
import math
import copy

def main():
    # puzzle_solver = PuzzleSolver("pokemon_1", (14.7, 10.5), 100, 100, 
    #     [('input/pokemon_puzzle_1_01.png', 20),
    #     ('input/pokemon_puzzle_1_02.png', 20),
    #     ('input/pokemon_puzzle_1_03.png', 20),
    #     ('input/pokemon_puzzle_1_04.png', 20),
    #     ('input/pokemon_puzzle_1_05.png', 20)], settings=[10, 50, 50, 14, 20, 128])

    # puzzle_solver = PuzzleSolver("pokemon_2", (15, 11), 100, 100, 
    #     [('input/pokemon_puzzle_2_01.png', 20),
    #     ('input/pokemon_puzzle_2_02.png', 20),
    #     ('input/pokemon_puzzle_2_03.png', 20),
    #     ('input/pokemon_puzzle_2_04.png', 20),
    #     ('input/pokemon_puzzle_2_05.png', 20)], settings=[10, 50, 50, 14, 20, 128])

    # puzzle_solver = PuzzleSolver("300", (21.25, 15), 100, 100,
    #     [('input/300_01.png', 30), ('input/300_02.png', 30), ('input/300_03.png', 30), ('input/300_04.png', 30),
    #     ('input/300_05.png', 30), ('input/300_06.png', 30), ('input/300_07.png', 30), ('input/300_08.png', 30), 
    #     ('input/300_09.png', 30), ('input/300_10.png', 30)], settings=[10, 25, 30, 12, 20, 32])

    puzzle_solver = PuzzleSolver("tart", (18, 18), 100, 200, 
        [('input/tart_puzzle_01.jpg', 30), ('input/tart_puzzle_02.jpg', 30), ('input/tart_puzzle_03.jpg', 30), ('input/tart_puzzle_04.jpg', 30),
        ('input/tart_puzzle_05.jpg', 30), ('input/tart_puzzle_06.jpg', 30), ('input/tart_puzzle_07.jpg', 28), ('input/tart_puzzle_08.jpg', 30),
        ('input/tart_puzzle_09.jpg', 30), ('input/tart_puzzle_10.jpg', 30), ('input/tart_puzzle_11.jpg', 26)], settings=[10, 50, 50, 12, 20, 32])

    # puzzle_solver = PuzzleSolver("travel", (21.25, 15), 100, 200,
    #     [('input/travel_puzzle_01.jpg', 30),('input/travel_puzzle_02.jpg', 30),('input/travel_puzzle_03.jpg', 30),('input/travel_puzzle_04.jpg', 30),
    #     ('input/travel_puzzle_05.jpg', 30),('input/travel_puzzle_06.jpg', 30),('input/travel_puzzle_07.jpg', 30),('input/travel_puzzle_08.jpg', 30),
    #     ('input/travel_puzzle_09.jpg', 30),('input/travel_puzzle_10.jpg', 12),('input/travel_puzzle_11.jpg', 18)], settings=[10, 50, 50, 12, 20, 32])
    
    # puzzle_solver = PuzzleSolver("shining", (18, 24), 200, 10000, 
    #     [('input/shining_01.jpg', 42), ('input/shining_02.jpg', 42), ('input/shining_03.jpg', 42), ('input/shining_04.jpg', 42),
    #     ('input/shining_05.jpg', 42), ('input/shining_06.jpg', 42), ('input/shining_07.jpg', 42), ('input/shining_08.jpg', 42),
    #     ('input/shining_09.jpg', 42), ('input/shining_10.jpg', 42), ('input/shining_11.jpg', 42), ('input/shining_12.jpg', 23),
    #     ('input/shining_13.jpg', 15)], settings=[10, 50, 50, 14, 20, 128])

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
        self.min_exp = 0.2
        self.max_exp = self.collection.num_pieces_total // 25
        self.exp_function = (lambda x:x) # ranges from 0 to 1

        self.edge_count_dict = {} # used in calculating similarity score

        self.solutions = None
        self.best_solution = None

        self.puzzle_dims = getPuzzleDims(dims, self.collection.num_pieces_total)
        self.dist_dict, self.sorted_dists, self.empty_edge_dist, _ = getDistDict(self.collection.pieces)

        self.doGen0(self.sides_first, set())

    
    def solvePuzzle(self):
        if self.sides_first:
            while(self.generation_counter < self.num_gens):
                self.doGeneration()
            self.min_exp = 0.2
            self.max_exp = self.collection.num_pieces_total // 25
            self.doGen0(False, self.best_solution.edges)
        while(self.generation_counter < self.num_gens):
            self.doGeneration()
        print(f'best score found in {self.num_gens} generations: {self.best_solution.score}')
        print(f'total time: {self.total_time}')
        solution_image = self.best_solution.getSolutionImage()

        h, w, _ = solution_image.shape
        cv2.destroyAllWindows()
        cv2.imshow('best solution', cv2.resize(solution_image, (int(500 * (w / h)), 500), interpolation=cv2.INTER_AREA))
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.imwrite(f'{self.puzzle_name}Solution.jpg', solution_image)
    
    def doGeneration(self):
        collection = copy.copy(self.collection) 
        if self.sides_first:
            collection.pieces = [piece for piece in collection.pieces if piece.type == 'side' or piece.type=='corner']

        gen_time = 0

        solutions = sorted(self.solutions, key=lambda x:x.score)

        top_10_percent = solutions[:self.gen_size // 10]

        limit = 10*self.gen_size
        counter = 0

        if self.generation_counter == 1:
            selection = random.choices(solutions, k=self.gen_size//2)
        else:
            selection = select_all(solutions, 4*self.gen_size//10, 2)
            selection += top_10_percent
            selection.append(self.best_solution.mutate(0))

        # new_solutions += selection
        # maybe keep top 10% of scores, select remaining 1/2, crossover to fill
        new_solutions = selection

        #new_solutions += top_percent
        while len(new_solutions) < self.gen_size:
            start = timer()
            parent1, parent2 = random.choices(selection, k=2)
            if abs(parent1.score - parent2.score) < 1: # to prevent infinite looping :(
                counter += 1
                if counter > limit:
                    solver = parent1.mutate(0.95) 
                else:
                    continue
            else:
                solver = parent1.crossover(parent2)
            
            new_solutions.append(solver)
            #print(f'gen {gen} solution {len(new_solutions)}')
            #print(f'p1 score: {parent1.score}, p2 score: {parent2.score}, c score: {solver.score}')
            end = timer()
            #print(f'time: {end - start}')
            gen_time += (end - start)

            #print('\n')
            if len(new_solutions) % 50 == 0:
                print(f'\t{len(new_solutions)}')

        start = timer()

        improved = False

        self.updateSimilarityScore(new_solutions)

        scores_before_mutation = [solution.score for solution in new_solutions]
        if min(scores_before_mutation) < self.best_solution.score:
            self.best_solution = min(new_solutions, key=lambda x:x.score).mutate(0) # copy
            improved = True

        print(f'top 10 scores gen {self.generation_counter} before mutation: {sorted(scores_before_mutation)[:10]}')
        solutions = [solution.mutate(self.getMutationRate(solution)) for solution in new_solutions]

        new_scores = [solution.score for solution in solutions]

        if min(new_scores) < self.best_solution.score:
            self.best_solution = min(solutions, key=lambda x:x.score).mutate(0)
            improved = True
        
        if improved:
            solution_image = self.best_solution.getSolutionImage()
            h, w, _ = solution_image.shape
            #cv2.imshow(f'best solution gen {gen}', cv2.resize(solution_image, (int(500 * (w / h)), 500), interpolation=cv2.INTER_AREA))
            #cv2.waitKey()
            cv2.imwrite(f'best_solution_gen_{self.generation_counter}_{self.puzzle_name}.jpg', solution_image)

        end = timer()
        gen_time += (end - start)
        print(f'total time gen {self.generation_counter}: {gen_time}')

        solutions = sorted(solutions, key=lambda x:x.score)
        scores = [int(solution.score) for solution in solutions]
        scores2 = sorted(list(set(scores)))
        diversities = sorted([solution.similarity_score for solution in solutions])

        self.solutions = solutions
        self.updateSimilarityScore(self.solutions)

        #top_percent = solutions[:len(solutions)//10]
        print(f'best score gen {self.generation_counter} : {self.best_solution.score}')
        print(f'top 10 gen {self.generation_counter} scores: {scores[:10]},\n top 10 unique: {scores2[:10]}')
        print(f'top 10 scores diversities: {[solution.similarity_score for solution in solutions[:10]]}')
        print(f'top 10 scores mutation rates: {[self.getMutationRate(solution) for solution in solutions[:10]]}')
        print(f'top 10 scores expected # mutations: {[1 / (1 - self.getMutationRate(solution)) - 1 for solution in solutions[:10]]}')
        print(f'average score: {sum(scores) // len(scores)}, num varieties: {len(scores2)}\n\n')

        self.generation_counter += 1
        self.total_time += gen_time

    def doGen0(self, just_sides, include_edges):
        collection = copy.copy(self.collection)
        solutions = []
        side_pieces = [piece for piece in collection.pieces if piece.type == 'side' or piece.type == 'corner']
        if just_sides:
            collection.pieces = [piece for piece in collection.pieces if piece.type == 'side' or piece.type=='corner']
            self.min_exp = 0
            self.max_exp = len(collection.pieces) // 25
        print('beggining genetic process')
        gen_time = 0
        for i in range(self.gen_size):
            start = timer()
            solver = PuzzleSolution(collection, self.puzzle_dims, self.dist_dict, self.sorted_dists, self.empty_edge_dist)
            if len(include_edges) > 0:
                side_edges = copy.copy(include_edges)
                for j in range(2):
                    side_edges.remove(random.choice(list(side_edges)))
                solver.solvePuzzle(start=random.choice(side_pieces), include_edges=side_edges)
            else:
                solver.solvePuzzle(random_start=True)
            solutions.append(solver)
            end = timer()
            gen_time += (end - start)

        self.best_solution = min(solutions, key=lambda x:x.score)
        self.solutions = solutions
        self.updateSimilarityScore(self.solutions)

        solution_image = self.best_solution.getSolutionImage()
        cv2.imwrite(f'best_solution_gen_0{self.puzzle_name}.jpg', solution_image)   
        print(f'total time gen 0: {gen_time}')        
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
        exp_fun = self.exp_function #(lambda x:x**3) #exp_fun should be a function that ranges btw 0 and 1
        
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
        for solver in solvers:
            for edge in solver.edges:
                count = self.edge_count_dict.get(edge)
                if count is None:
                    count = 0
                self.edge_count_dict[edge] = count + 1

        for solver in solvers:
            solver.similarity_score = 0   
            for edge in solver.edges:
                num_edges = self.edge_count_dict.get(edge)
                solver.similarity_score += (num_edges - 1)
            solver.similarity_score /= (self.gen_size * len(solver.edges) * (self.generation_counter + 1))
        # if an edge was in every single solution in every generation, then it would add 1 to the diversity score
        # if an edge was in NO others then it would add 1/(gen_size*(gen_counter + 1)) to the total

        # "the average edge in (solver) is in (100 * similarity_score) % of the population seen so far"

        # so the diversity score ranges from 1 / num_edges (if all unique) to 1 if all in every other solution

def select_all(solvers, n, k):
    selection = []
    for i in range(n):
        selection.append(select_one_by_score(solvers, k))
    return selection

def select_one_by_score(solvers, k):
    random_solvers = random.choices(solvers, k=k)
    best = min(random_solvers, key=lambda x:x.score * ( 1 + x.similarity_score**2))
    return best

# temporary filler function
def getPuzzleDims(size_inches, num_pieces):
    diff = 1000
    side_Ratio = size_inches[1] / size_inches[0]
    predict_Ratio = 0
    predict_w,predict_h = 0,0

    for i in range(1,int(pow(num_pieces, 1 / 2)) + 1):
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
    return predict_h,predict_w # for now just putting the actual number of pieces in as size_inches oops

# returns a dict containing the distances between all edges
def getDistDict(pieces):
    dist_dict = {}
    sorted_dists = {}
    max_dist = 0
    for piece1 in pieces:
        print(piece1.label)
        for edge1 in range(4):
            dist_dict[(piece1, edge1)] = {}
        for edge1 in range(4):
            dist_list = []
            for piece2 in pieces:
                for edge2 in range(4):
                    if piece1 == piece2:
                        dist = float('inf')
                    else:
                        dist = piece1.edges[edge1].compare(piece2.edges[edge2])
                    dist_dict[(piece1, edge1)][(piece2, edge2)] = dist
                    if dist < float('inf'):
                        if dist > max_dist:
                            max_dist = dist
                        dist_list.append(((piece2, edge2), dist))
            sorted_dists[(piece1, edge1)] = sorted(dist_list, key=lambda x:x[1])
    return dist_dict, sorted_dists, max_dist, float('inf')



if __name__ == "__main__":
    main()