from email.policy import default
import pygame
import time
import pathlib
import button
import fileBox
from tkinter import filedialog


def shortenPath(path, new_len):
    return pathlib.Path(*pathlib.Path(path).parts[-new_len:]).__str__()



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
        self.x = 640
        self.y = 480
        self.resultImage = 'test.png'

    # Create a window and set its needed settings
    def initWindow(self):
        # Set title
        pygame.display.set_caption(self.title)
        # Create drawing surface
        self.window = pygame.display.set_mode([self.width, self.height])
        self.curent_add_point = 0
        self.paths = []

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
            # self.resultImage = 'result.jpg'
            print('The run button was clicked!')

        # # Create button object
        # add = button.Button('+', 10, 60, 230, 30, f2,
        #                     off_white, button_selected, button_unselected)
        # # draw the button
        # add.draw(self.window)
        # # Handle the click
        # if self.clicked and add.isInButton(self.last_click_x, self.last_click_y):
        #     self.clicked = False
        #     print('Add button was clicked!')

        if self.curent_add_point < 13:
            add_button = button.Button('+', 10, 60 + (40 * self.curent_add_point),
                                       230, 30, f2, off_white, button_selected, button_unselected)
            add_button.draw(self.window)
            if self.clicked and add_button.isInButton(self.last_click_x, self.last_click_y):
                self.clicked = False
                path = filedialog.askopenfilename(
                    filetypes=[('.png', '*.png')])
                if path != '':
                    self.curent_add_point = self.curent_add_point + 1
                    self.paths.append(path)

        # Temp file box
        # file_null = fileBox.FileBox('NULL', 10, 100, 230, 30, f2)
        # file_null.draw(self.window)
        # if self.clicked and file_null.isInButton(self.last_click_x, self.last_click_y):
        #     self.clicked = False
        #     print('X pressed')

        offset = 0
        for path in self.paths:
            file_box = fileBox.FileBox(
                shortenPath(path, 1), 10, 60 + (40 * offset), 230, 30, f2)
            file_box.draw(self.window)
            offset = offset + 1
            if self.clicked and file_box.isInButton(self.last_click_x, self.last_click_y):
                self.clicked = False
                self.paths.remove(path)
                self.curent_add_point = self.curent_add_point - 1

    # Render in the result of simpleSolver and Zoom button
    def drawSolverArea(self):
        #Definitions for Colors and Fonts of Various Buttons. Same as UploadRegion for consistency
        std_font = pygame.font.Font(pygame.font.get_default_font(), 48)
        off_white = (230, 230, 230)
        button_selected = (80, 80, 80)
        button_unselected = (50, 50, 50)


        #Implementation of Solved Puzzle Section (Puzzle solver output will display using this.)
        testIMG = pygame.image.load(self.resultImage)
        testIMG = pygame.transform.scale(testIMG, (self.x, self.y))
        self.window.blit(testIMG, (300, 46))
        

        #Implementation of Zoom In Button
        zoomPlus = button.Button(
            '+', ((self.window.get_width() / 2) + 40), 590, 50, 50, 
                    std_font, off_white, button_selected, button_unselected)
        zoomPlus.draw(self.window)
        if self.clicked and zoomPlus.isInButton(self.last_click_x, self.last_click_y):
            self.clicked = False
            self.x = self.x + 60 
            self.y = self.y + 60
            #self.window.blit(testIMG, (300, 46))
            print("Zooming into image")

        #Implementation for Zoom Out Button
        zoomMinus = button.Button(
            '-', (self.window.get_width() / 2) -40, 590, 50, 50, 
                    std_font, off_white, button_selected, button_unselected)
        zoomMinus.draw(self.window)
        if self.clicked and zoomMinus.isInButton(self.last_click_x, self.last_click_y):
            self.clicked = False
            self.x = self.x - 60 
            self.y = self.y - 60
            #self.window.blit(testIMG, (300, 46))
            print("Zooming out from image")




    # Render stuff to the window

    def render(self):
        # Clear screen with white
        self.window.fill((255, 253, 231))

        # DRAW HERE (Things are rended from top to bottom, last listed is top layer)

        self.drawUploadRegion()

        self.drawSolverArea()

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
