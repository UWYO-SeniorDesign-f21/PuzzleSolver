import pygame
from window import Window

if __name__ == '__main__':
    # Initialize the library
    pygame.init()
    # Create a new window object
    window_ = Window("Puzzle Solver", 1000, 650)
    # Call the run method
    window_.run()
