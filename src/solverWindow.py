from email.policy import default
from re import U
from turtle import pos
from xmlrpc.client import Boolean
from pieceCollection import showLabels
from Text_box_class import *
import pygame
import time
import pathlib
import button
import fileBox
from tkinter import filedialog
import os
vec = pygame.math.Vector2

text_boxes = []


def shortenPath(path, new_len):
    return pathlib.Path(*pathlib.Path(path).parts[-new_len:]).__str__()


class Text_box:
    def __init__(self, x, y, width, height, bg_colour=(124, 124, 124), active_colour=(255, 255, 255)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.pos = vec(x, y)
        self.size = vec(width, height)
        self.image = pygame.Surface((width, height))
        self.bg_colour = bg_colour
        self.active_colour = active_colour
        self.active = False
        self.text = ""

    def drawTextBox(self, window):
        if not self.active:
            self.image.fill(self.bg_colour)
        else:
            self.image.fill(self.active_colour)

        window.blit(self.image, self.pos)

    def add_text(self, key):
        print(key)

    def checkTextClick(self, pos):
        if pos[0] > self.x and pos[0] < self.x + self.width:
            if pos[1] > self.y and pos[1] < self.y + self.height:
                self.active = True
            else:
                self.active = False
        else:
            self.active = False


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
        self.centerScreenx = (width / 2) + 150
        self.centerScreeny = height / 2
        self.resultImage = 'result.jpg'
        self.text_boxes = []

    # Create a window and set its needed settings
    def initWindow(self):
        # Set title
        pygame.display.set_caption(self.title)
        # Create drawing surface
        self.window = pygame.display.set_mode([self.width, self.height])
        self.curent_add_point = 0
        self.paths = []
        self.settings = False
        self.closeClick = False
        text_boxes.append(
            Text_box(((self.window.get_width() / 3 * 2)+50), 60, 70, 40))
        text_boxes.append(
            Text_box(((self.window.get_width() / 3 * 2)+150), 60, 70, 40))

    # Process all events on the window

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Tells loop to stop
                pygame.quit()
                self.settings = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                # get mouse position
                mouse = pygame.mouse.get_pos()
                # setup mouse for event handling -> MAKE CERTAIN TO SET self.clicked BACK TO FALSE WHEN EVENT HAS BEEN PROCESSED
                self.last_click_x = mouse[0]
                self.last_click_y = mouse[1]

                self.clicked = True
            if event.type == pygame.KEYDOWN:
                for box in text_boxes:
                    if box.active:
                        box.add_text(event.key)

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
        # Definitions for Colors and Fonts of Various Buttons. Same as UploadRegion for consistency
        std_font = pygame.font.Font(pygame.font.get_default_font(), 48)
        off_white = (230, 230, 230)
        button_selected = (80, 80, 80)
        button_unselected = (50, 50, 50)

        # Implementation of Solved Puzzle Section (Puzzle solver output will display using this.)
        testIMG = pygame.image.load(self.resultImage)
        testIMG.convert()
        testIMG = pygame.transform.scale(testIMG, (self.x, self.y))
        testBorder = testIMG.get_rect()
        testBorder.center = self.centerScreenx, self.centerScreeny
        self.window.blit(testIMG, testBorder)
        MPOS = pygame.mouse.get_pos()

        # Maybe an Implementation of Mouse Panning (Still in the works.)
        # if self.clicked and MPOS >= (testBorder.x, testBorder.y):
        #MPOS = pygame.mouse.get_pos()
        #print("Trying to Pan Image")
        #self.centerScreenx = MPOS[0]
        #self.centerScreeny = MPOS[1]
        #self.clicked = False
        # print(MPOS)

    def drawAllTheButtons(self):
        # Definitions for Colors and Fonts of Various Buttons. Same as UploadRegion for consistency
        std_font = pygame.font.Font(pygame.font.get_default_font(), 48)
        off_white = (230, 230, 230)
        button_selected = (80, 80, 80)
        button_unselected = (50, 50, 50)

        # Implementation of Zoom In Button
        zoomPlus = button.Button(
            '+', 250, 550, 50, 50,
            std_font, off_white, button_selected, button_unselected)
        zoomPlus.draw(self.window)
        if self.clicked and zoomPlus.isInButton(self.last_click_x, self.last_click_y):
            self.clicked = False
            self.x = self.x + 80
            self.y = self.y + 60

        # Implementation for Zoom Out Button
        zoomMinus = button.Button(
            '-', (250), 600, 50, 50,
            std_font, off_white, button_selected, button_unselected)
        zoomMinus.draw(self.window)
        if self.clicked and zoomMinus.isInButton(self.last_click_x, self.last_click_y):
            self.clicked = False
            if (self.x - 120 <= 0 or self.y - 120 <= 0):
                print("Cannot zoom out further")
                self.x = 60
                self.y = 60
            else:
                self.x = self.x - 80
                self.y = self.y - 60

        # Right Button Implementation
        goRight = button.Button(
            u"->",  (self.width-50), (self.height/2)-50, 50, 50,
            std_font, off_white, button_selected, button_unselected)
        goRight.draw(self.window)
        if self.clicked and goRight.isInButton(self.last_click_x, self.last_click_y):
            self.clicked = False
            self.centerScreenx = self.centerScreenx + 50
            #self.window.blit(testIMG, (300, 46))
            print("Panning to Right?")

        # Left Button Implementation
        goLeft = button.Button(
            u"<-", (self.width/4), (self.height/2)-50, 50, 50,
            std_font, off_white, button_selected, button_unselected)
        goLeft.draw(self.window)
        if self.clicked and goLeft.isInButton(self.last_click_x, self.last_click_y):
            self.clicked = False
            self.centerScreenx = self.centerScreenx - 50
            #self.window.blit(testIMG, (300, 46))
            print("Panning to Left?")

        # Up Button Implementation
        goUp = button.Button(
            "^", (self.width/2)-50, 0, 50, 50,
            std_font, off_white, button_selected, button_unselected)
        goUp.draw(self.window)
        if self.clicked and goUp.isInButton(self.last_click_x, self.last_click_y):
            self.clicked = False
            self.centerScreeny = self.centerScreeny - 50
            print("Panning Up?")

        # Implements Down Button
        goDown = button.Button(
            ('\u2193'), (self.width/2)-50, (self.height-50), 50, 50,
            std_font, off_white, button_selected, button_unselected)
        goDown.draw(self.window)
        if self.clicked and goDown.isInButton(self.last_click_x, self.last_click_y):
            self.clicked = False
            self.centerScreeny = self.centerScreeny + 50
            print("Panning Down?")

        # Access settings.png from current directory
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        my_file = os.path.join(THIS_FOLDER, 'settings.png')

        photo = pygame.image.load(my_file)

        # Scale the image to size
        DEFAULT_IMAGE_SIZE = (55, 55)
        settingsClicked = False

        photo = pygame.transform.scale(photo, DEFAULT_IMAGE_SIZE)

        # Create button
        settingsButton = button.Button(
            '', self.window.get_width() - 60, self.window.get_height() - 60, 55, 55,
            std_font, off_white, (210, 210, 210), (240, 240, 240))
        settingsButton.draw(self.window)
        if self.clicked and settingsButton.isInButton(self.last_click_x, self.last_click_y):
           # self.clicked = False
            pygame.draw.rect(self.window, (150, 150, 150), pygame.Rect(
                2 * (self.window.get_width() / 3), 0, self.window.get_width(), self.window.get_height()))

        self.window.blit(photo, (self.window.get_width() -
                         60, self.window.get_height()-60))

    def drawSettingsButton(self):
        std_font = pygame.font.Font(pygame.font.get_default_font(), 48)
        off_white = (230, 230, 230)
        button_selected = (80, 80, 80)
        button_unselected = (50, 50, 50)

        # Access settings.png from current directory
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        my_file = os.path.join(THIS_FOLDER, 'settings.png')

        photo = pygame.image.load(my_file)

        # Scale the image to size
        DEFAULT_IMAGE_SIZE = (55, 55)
        settingsClicked = False

        settingsWheel = pygame.transform.scale(photo, DEFAULT_IMAGE_SIZE)

        # Create button
        settingsButton = button.Button(
            '', self.window.get_width() - 60, self.window.get_height() - 60, 55, 55,
            std_font, off_white, (210, 210, 210), (240, 240, 240))
        settingsButton.draw(self.window)
        if self.clicked and settingsButton.isInButton(self.last_click_x, self.last_click_y):
            self.settings = True

        self.window.blit(settingsWheel, (self.window.get_width() -
                         60, self.window.get_height()-60))

        # create settings close button
    def drawSettingsClose(self):
        std_font = pygame.font.Font(pygame.font.get_default_font(), 48)
        off_white = (230, 230, 230)
        button_selected = (80, 80, 80)
        button_unselected = (200, 200, 200)

        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        my_file = os.path.join(THIS_FOLDER, 'closeX.png')

        DEFAULT_IMAGE_SIZE = (55, 55)

        closeX = pygame.image.load(my_file)

        closeX = pygame.transform.scale(closeX, DEFAULT_IMAGE_SIZE)

        if self.settings:
            settingsClose = button.Button(
                'x', self.window.get_width() - 60, 0, 55, 55,
                std_font, off_white, (210, 210, 210), (240, 240, 240))

        self.window.blit(closeX, (self.window.get_width() -
                         60, 0))

        if self.clicked and settingsClose.isInButton(self.last_click_x, self.last_click_y):
            self.closeClick = True

    def drawSettingsScreen(self):
        # nothin yet
        x = 1
        pygame.draw.rect(self.window, (50, 50, 50), pygame.Rect(2 * (self.window.get_width() / 3),
                                                                0, self.window.get_width(), self.window.get_height()))

    def drawTextBoxes(self):
        if self.clicked:
            for box in text_boxes:
                box.checkTextClick((self.last_click_x, self.last_click_y))

        for box in text_boxes:
            box.drawTextBox(self.window)

    def render(self):
        # Clear screen with white
        self.window.fill((255, 253, 231))
        i = 1

        # DRAW HERE (Things are rended from top to bottom, last listed is top layer)

        if not self.settings:
            self.drawSolverArea()
            self.drawUploadRegion()
            self.drawAllTheButtons()
            self.drawSettingsButton()

        else:
            self.drawSolverArea()
            self.drawUploadRegion()
            self.drawSettingsScreen()
            self.drawSettingsClose()
            self.drawTextBoxes()

            if self.closeClick:
                self.settings = False
                self.drawUploadRegion()
                self.drawSolverArea()
                self.closeClick = False

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
                self.render()
                self.update()
                if not self.running:
                    break

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
