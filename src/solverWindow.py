import pygame
import time


class Window:
    # Constructor sets all nessasary variables for creating a window
    def __init__(self, title: str, width: int, height: int):
        self.title = title
        self.width = width
        self.height = height
        self.running = False

    # Create a window and set its needed settings
    def initWindow(self):
        # Set title
        pygame.display.set_caption(self.title)
        # Create drawing surface
        self.window = pygame.display.set_mode([self.width, self.height])

    # Process all events on the window

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    # Used to not clutter the render method
    def drawUploadRegion(self):
        # Draw a colored rectagle to differentiate the region
        pygame.draw.rect(self.window, (74, 74, 74), pygame.Rect(
            0, 0, self.window.get_width() / 4, self.window.get_height()))

    # Render stuff to the window
    def render(self):
        # Clear screen with white
        self.window.fill((255, 255, 255))

        # DRAW HERE (Things are rended from top to bottom)

        self.drawUploadRegion()

        # END DRAWING

        # Display updated render to screen
        pygame.display.flip()

    # Runs the window
    def run(self):
        self.running = True
        self.initWindow()

        # Set run loop to operate at 72 fps
        fps = 72
        timePerFrame = 1000000000 / fps
        delta = 1.0
        now = int()
        last = time.time_ns()
        while self.running:
            now = time.time_ns()
            delta = delta + ((now - last) / timePerFrame)
            last = now
            if delta >= 1:
                # Call update and render if on time
                self.update()
                self.render()
                delta = delta - 1
        # Quit pygame when loop completes
        pygame.quit()


if __name__ == '__main__':
    # Initialize the library
    pygame.init()
    # Create a new window object
    window = Window("Puzzle Solver", 1000, 650)
    # Call the run method
    window.run()
