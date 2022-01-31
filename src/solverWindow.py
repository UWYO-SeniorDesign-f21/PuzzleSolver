import pygame
import time
import button
import fileBox


class Window:
    # Constructor sets all nessasary variables for creating a window
    def __init__(self, title: str, width: int, height: int):
        self.title = title
        self.width = width
        self.height = height
        self.running = False
        self.last_click_x = int()
        self.last_click_y = int()
        self.clicked = False

    # Create a window and set its needed settings
    def initWindow(self):
        # Set title
        pygame.display.set_caption(self.title)
        # Create drawing surface
        self.window = pygame.display.set_mode([self.width, self.height])
        self.curent_add_point = 0

    # Process all events on the window

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Tells loop to stop
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # get mouse position
                mouse = pygame.mouse.get_pos()
                # setup mouse for event handling -> MAKE CERTAIN TO SET self.clicked BACK TO FALSE WHEN EVENT HAS BEEN PROCESSED
                self.last_click_x = mouse[0]
                self.last_click_y = mouse[1]
                self.clicked = True

    # Used to not clutter the render method
    def drawUploadRegion(self):
        # set some frequently used colors
        bg_gray = (74, 74, 74)
        off_white = (230, 230, 230)

        # Draw a colored rectagle to differentiate the region
        pygame.draw.rect(self.window, bg_gray, pygame.Rect(
            0, 0, self.window.get_width() / 4, self.window.get_height()))

        # Labels for upload area
        # Configure font
        f1 = pygame.font.Font(pygame.font.get_default_font(), 34)

        # Setup label
        mLabel = f1.render('Upload Pieces', True, off_white, bg_gray)
        mLabel_rect = mLabel.get_rect()
        mLabel_rect.topleft = (5, 5)

        self.window.blit(mLabel, mLabel_rect)

        # Creating add file Button
        # set some colors for weather or not the button is selected
        button_selected = (60, 60, 60)
        button_unselected = (50, 50, 50)
        # set the buttons font
        f2 = pygame.font.Font(pygame.font.get_default_font(), 12)

        # Run Button stuff
        run_button = button.Button(
            'Run', 10, 610, 230, 30, f2, off_white, button_selected, button_unselected)
        run_button.draw(self.window)
        if self.clicked and run_button.isInButton(self.last_click_x, self.last_click_y):
            self.clicked = False
            print('The run button was clicked!')

        # Create button object
        add = button.Button('+', 10, 60, 230, 30, f2,
                            off_white, button_selected, button_unselected)
        # draw the button
        add.draw(self.window)
        # Handle the click
        if self.clicked and add.isInButton(self.last_click_x, self.last_click_y):
            self.clicked = False
            print('Add button was clicked!')

        # Temp file box
        # file_null = fileBox.FileBox('NULL', 10, 100, 230, 30, f2)
        # file_null.draw(self.window)
        # if self.clicked and file_null.isInButton(self.last_click_x, self.last_click_y):
        #     self.clicked = False
        #     print('X pressed')

    # Render stuff to the window
    def render(self):
        # Clear screen with white
        self.window.fill((255, 255, 255))

        # DRAW HERE (Things are rended from top to bottom, last listed is top layer)

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
