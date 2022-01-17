from solutionMiddle import Solution
from pieceCollection import PieceCollection
import random
import cv2
import numpy as np

class PuzzleSolver:
    def __init__(self, pieces, size_inches, gen_size, num_gens):
        self.pieces = pieces
        self.puzzle_dims = getPuzzleDims(size_inches, len(pieces.pieces))
        self.gen_size = gen_size
        self.num_gens = num_gens
        self.gen_counter = 0
        self.dist_dict = getDistDict(self.pieces.pieces)
        self.buddy_dict = getBuddyDict(self.pieces.pieces, self.dist_dict)
        self.initializeGenZero()

    def initializeGenZero(self):
        self.solutions = []
        for i in range(self.gen_size):
            random_solution = Solution(self.pieces, self.puzzle_dims, self.dist_dict, self.buddy_dict)
            random_solution.randomize()
            print(f'\t solution {i}, score {random_solution.score}')
            self.solutions.append(random_solution)

    def doGeneration(self):
        parent_solutions = sorted(self.solutions, key=lambda x:x.score)
        parent_solutions = parent_solutions[:len(parent_solutions) // 5]
        scores = [solution.score for solution in parent_solutions]
        scores = (np.max(scores) - scores) / (np.max(scores) - np.min(scores)) # 1 - (scores / np.linalg.norm(scores))
        new_solutions = []
        counter = 0
        while len(new_solutions) < self.gen_size:
            random_solutions = random.choices(parent_solutions, k=2)
            new_solution = random_solutions[0].crossover(random_solutions[1])
            print(f'\tsolution {counter}, parents scores: {random_solutions[0].score, random_solutions[1].score}, score {new_solution.score}')
            counter += 1
            new_solutions.append(new_solution)
        self.solutions = new_solutions
        self.gen_counter += 1

    def doRemainingGens(self, show_images=False):
        while self.gen_counter < self.num_gens:
            self.doGeneration()
            if show_images:
                best_solution = self.getBestSolution()
                solution_image = best_solution.getSolutionImage()
                print(f'best score: {best_solution.score}')
                h, w, _ = solution_image.shape
                cv2.imshow(f'best solution gen {self.gen_counter}', cv2.resize(solution_image, (int(500 * (w / h)), 500), interpolation=cv2.INTER_AREA))
                cv2.waitKey(10)
        if show_images:
            cv2.waitKey()

    def getBestSolution(self):
        best = min(self.solutions, key=lambda x:x.score)
        return best
    
def getPuzzleDims(size_inches, num_pieces):
    return 10, 10

def getDistDict(pieces):
    dist_dict = {}
    for piece1 in pieces:
        for piece2 in pieces:
            if piece1 == piece2:
                continue
            for edge1 in range(4):
                for edge2 in range(4):
                    dist = piece1.edges[edge1].compare(piece2.edges[edge2])
                    dist_dict[(piece1, edge1, piece2, edge2)] = dist
    return dist_dict

def getBuddyDict(pieces, dist_dict):
    closest_pieces_dict = {}
    buddy_dict = {}
    for piece1 in pieces:
        for edge1 in range(4):
            min_edge = None
            min_cost = float('inf')
            for piece2 in pieces:
                if piece1 == piece2:
                    continue
                for edge2 in range(4):
                    dist = dist_dict[(piece1, edge1, piece2, edge2)]
                    if dist < min_cost:
                        min_cost = dist
                        min_edge = (piece2, edge2)
            if min_edge:
                closest_pieces_dict[(piece1, edge1)] = min_edge
    for key in closest_pieces_dict.keys():
        val = closest_pieces_dict[key]
        if closest_pieces_dict.get(val) == key:
            buddy_dict[key] = val
            buddy_dict[val] = key
    return buddy_dict



if __name__=='__main__':
    collection = PieceCollection()
    collection.addPieces('puzzle3_02.jpg', 20)
    collection.addPieces('puzzle3_01.jpg', 20)
    collection.addPieces('puzzle3_03.jpg', 20)
    collection.addPieces('puzzle3_04.jpg', 20)
    collection.addPieces('puzzle3_05.jpg', 20)

    solver = PuzzleSolver(collection, (10, 10), 50, 50)
    best_solution_gen1 = solver.getBestSolution()
    solution_image = best_solution_gen1.getSolutionImage()
    print(f'best score: {best_solution_gen1.score}')
    h, w, _ = solution_image.shape
    cv2.imshow('best solution gen 0', cv2.resize(solution_image, (int(500 * (w / h)), 500), interpolation=cv2.INTER_AREA))
    cv2.waitKey(10)

    solver.doRemainingGens(show_images=True)
