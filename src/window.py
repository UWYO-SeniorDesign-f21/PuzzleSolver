from turtle import right
from numpy import true_divide
import pygame
import fileBox
import os
import button
import time
import pathlib
from tkinter import filedialog
from puzzleSolver import PuzzleSolver
from textBox import Text_box
from threading import Thread
import json

vec = pygame.math.Vector2

text_boxes = []


def shortenPath(path, new_len):
    return pathlib.Path(*pathlib.Path(path).parts[-new_len:]).__str__()


def runSolver(width_txt, height_txt, gen_txt, size_txt, paths):
    print("running solver...")
    solver = PuzzleSolver(".title", (int(width_txt), int(
        height_txt)), int(gen_txt), int(size_txt), paths, show_sols=False)
    solver.solvePuzzle_gui_mode()

def runSolverJSON(path):
    with open(path) as f:
        puzzle_data = json.load(f)
        print("running solver...")
        #print([entry for entry in puzzle_data["file_info"]])
        file_list = [('../' + entry["path"], entry["num_pieces"]) for entry in puzzle_data["file_info"]]
        solver = PuzzleSolver(puzzle_data["puzzle_name"], tuple(puzzle_data["dims"]), puzzle_data["num_gens"],
                puzzle_data["gen_size"], file_list,
                color_spec = puzzle_data["color_spec"], show_sols=False)
    solver.solvePuzzle_gui_mode()

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
        self.mouseOverButton = False
        self.drag_click = False
        self.drag_click_pos = None
        self.drag_click_time = None
        self.key_pressed = None
        self.zoom_in = False
        self.zoom_out = False
        self.x = 640
        self.y = 480
        self.x_step = self.x // 10
        self.y_step = self.y // 10
        self.centerScreenx = (width / 2) + 150
        self.centerScreeny = height / 2
        self.resultImage = 'result.jpg'
        self.text_boxes = []

        self.json_file = None

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
            Text_box(((self.window.get_width() / 3 * 2) + 30), 53, 66, 22, border=2))
        text_boxes.append(
            Text_box(((self.window.get_width() / 3 * 2)+103), 53, 66, 22, border=2))
        text_boxes.append(
            Text_box(((self.window.get_width() / 3 * 2) + 30), (93 + 53), 66, 22, border=2))
        text_boxes.append(
            Text_box(((self.window.get_width() / 3 * 2) + 30), (93 * 2 + 53), 66, 22, border=2))

    # Process all events on the window

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Tells loop to stop
                if os.path.exists(".titleSolution.jpg"):
                    os.remove(".titleSolution.jpg")
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 4:
                    self.zoom_in = True
                elif event.button == 5:
                    self.zoom_out = True
                elif event.button == 1:
                    testIMG = pygame.image.load(self.resultImage)
                    testIMG.convert()
                    testIMG = pygame.transform.scale(testIMG, (self.x, self.y))
                    testBorder = testIMG.get_rect()
                    testBorder.center = self.centerScreenx, self.centerScreeny
                    self.window.blit(testIMG, testBorder)
                    MPOS = pygame.mouse.get_pos()

                    self.drag_click = True
                    self.drag_click_pos = MPOS
                    self.drag_click_time = time.time()

                # # Maybe an Implementation of Mouse Panning (Still in the works.)
                # if MPOS[0] >= (testBorder.x) and MPOS[0] <= (testBorder.x + self.x) and MPOS[1] >= (testBorder.y) and MPOS[1] <= (testBorder.y + self.y) and self.mouseOverButton == False:
                #     MPOS = pygame.mouse.get_pos()
                #     # print("Trying to Pan Image")
                #     self.centerScreenx = MPOS[0]
                #     self.centerScreeny = MPOS[1]
                #     print(MPOS)

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse = pygame.mouse.get_pos()
                    # setup mouse for event handling -> MAKE CERTAIN TO SET self.clicked BACK TO FALSE WHEN EVENT HAS BEEN PROCESSED
                    self.last_click_x = mouse[0]
                    self.last_click_y = mouse[1]

                    self.clicked = True
                    self.drag_click = False
                    self.drag_click_pos = None
                    self.drag_click_time = None
            if event.type == pygame.KEYDOWN:
                self.key_pressed = event.key
                for box in text_boxes:
                    if box.active:
                        box.add_text(event.key)
            if event.type == pygame.KEYUP:
                self.key_pressed = None

        if not self.drag_click_pos is None:
            if time.time() - self.drag_click_time >= 0.05:
                MPOS = pygame.mouse.get_pos()
                dx = self.drag_click_pos[0] - MPOS[0]
                dy = self.drag_click_pos[1] - MPOS[1]
                self.centerScreenx = self.centerScreenx - dx
                self.centerScreeny = self.centerScreeny - dy

                self.drag_click = True
                self.drag_click_pos = MPOS



    def prender(self):
        bg_gray = (74, 74, 74)
        off_white = (230, 230, 230)

        self.window.fill(bg_gray)

        f1 = pygame.font.Font(pygame.font.get_default_font(), 34)
        mLabel = f1.render('Enter Piece Count', True, off_white, bg_gray)
        mLabel_rect = mLabel.get_rect()
        mLabel_rect.topleft = (5, 5)

        if self.clicked:
            self.tb.checkTextClick((self.last_click_x, self.last_click_y))
        self.tb.drawTextBox(self.window)

        button_selected = (60, 60, 60)
        button_unselected = (50, 50, 50)
        # set the buttons font
        f2 = pygame.font.Font(pygame.font.get_default_font(), 12)

        # Run Button stuff
        run_button = button.Button(
            'Ok', 170, 100, 60, 30, f2, off_white, button_selected, button_unselected)
        run_button.draw(self.window)
        if self.clicked and run_button.isInButton(self.last_click_x, self.last_click_y):
            self.clicked = False
            self.pieces_count = int(self.tb.text)

        self.window.blit(mLabel, mLabel_rect)

        pygame.display.flip()

    def popup_mode(self):
        self.window = pygame.display.set_mode([311, 140])
        self.tb = Text_box(25, 50, 250, 30)
        self.pieces_count = -1

        fps = 72
        timePerFrame = 1000000000 / fps
        delta = 1.0
        now = int()
        last = time.time_ns()
        while True:
            now = time.time_ns()
            delta = delta + ((now - last) / timePerFrame)
            last = now

            if self.pieces_count != -1:
                break

            if delta >= 1:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    # elif event.type == pygame.MOUSEBUTTONDOWN:
                        # get mouse position
                        # mouse = pygame.mouse.get_pos()
                        # self.centerScreenx = mouse[0]
                        # self.centerScreeny = mouse[1]

                    elif event.type == pygame.MOUSEBUTTONUP:
                        mouse = pygame.mouse.get_pos()
                        # setup mouse for event handling -> MAKE CERTAIN TO SET self.clicked BACK TO FALSE WHEN EVENT HAS BEEN PROCESSED
                        self.last_click_x = mouse[0]
                        self.last_click_y = mouse[1]
                        self.clicked = True
                    elif event.type == pygame.KEYDOWN:
                        if self.tb.active:
                            self.tb.add_text(event.key)
                self.prender()
                delta = delta - 1
        self.window = pygame.display.set_mode([self.width, self.height])
        return self.pieces_count

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
            width_txt = text_boxes[0].text
            height_txt = text_boxes[1].text
            gen_txt = text_boxes[2].text
            size_txt = text_boxes[3].text

            if self.json_file is None and (width_txt == "" or height_txt == ""):
                print("Width and height must be set befor running!")
                return
            if gen_txt == "":
                gen_txt = "100"
            if size_txt == "":
                size_txt = "100"

            if self.json_file is None:
                new_thread = Thread(target=runSolver, args=(
                    width_txt, height_txt, gen_txt, size_txt, self.paths))
            else:
                new_thread = Thread(target=runSolverJSON, args=(self.json_file,))

            new_thread.start()

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
                    filetypes=[('JSON', '*.JSON'), ('jpg', '*.jpg'), ('png', '*.png')])
                if path.endswith('.JSON'):
                    self.json_file = path
                if path != '':
                    if self.json_file is None: 
                        pieces = self.popup_mode()
                        self.paths.append((path, pieces))
                    else:
                        self.paths.append((path, 0))
                    self.curent_add_point = self.curent_add_point + 1

        # Temp file box
        # file_null = fileBox.FileBox('NULL', 10, 100, 230, 30, f2)
        # file_null.draw(self.window)
        # if self.clicked and file_null.isInButton(self.last_click_x, self.last_click_y):
        #     self.clicked = False
        #     print('X pressed')

        offset = 0
        for path in self.paths:
            file_box = fileBox.FileBox(
                shortenPath(path[0], 1), 10, 60 + (40 * offset), 230, 30, f2)
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

    def drawAllTheButtons(self):
        # Definitions for Colors and Fonts of Various Buttons. Same as UploadRegion for consistency
        ARR_FOLDER = os.path.dirname(os.path.abspath(__file__))
        rightA = os.path.join(ARR_FOLDER, 'arrowRight.png')
        arrow = pygame.image.load(rightA)
        aIcon = pygame.transform.scale(arrow, (50, 50))
        std_font = pygame.font.Font(pygame.font.get_default_font(), 48)
        off_white = (230, 230, 230)
        button_selected = (80, 80, 80)
        button_unselected = (50, 50, 50)

        # Implementation of Zoom In Button
        zoomPlus = button.Button(
            '+', 250, 550, 50, 50,
            std_font, off_white, button_selected, button_unselected)
        zoomPlus.draw(self.window)
        if (self.clicked and zoomPlus.isInButton(self.last_click_x, self.last_click_y)) or self.zoom_in:
            self.zoom_in = False
            self.clicked = False
            self.x = int(self.x * 1.2) # + self.x_step
            self.y = int(self.y * 1.2) # + self.y_step

        # Implementation for Zoom Out Button
        zoomMinus = button.Button(
            '-', (250), 600, 50, 50,
            std_font, off_white, button_selected, button_unselected)
        zoomMinus.draw(self.window)
        if (self.clicked and zoomMinus.isInButton(self.last_click_x, self.last_click_y)) or self.zoom_out:
            self.zoom_out = False
            self.clicked = False
            if (self.x - 120 <= 0 or self.y - 120 <= 0):
                print("Cannot zoom out further")
                # self.x = 60
                # self.y = 60
            else:
                self.x = int(self.x * 0.8) # - self.x_step
                self.y = int(self.y * 0.8) # - self.y_step

        # Right Button Implementation
        goRight = button.Button(
            '',  (self.width-35), (self.height/2)-75, 35, 100,
            std_font, off_white, button_selected, button_unselected)
        goRight.draw(self.window)
        if (self.clicked and goRight.isInButton(self.last_click_x, self.last_click_y)) or self.key_pressed == pygame.K_RIGHT:
            self.mouseOverButton = True
            self.clicked = False
            self.centerScreenx = self.centerScreenx + 50
            #self.window.blit(testIMG, (300, 46))
            print("Panning to Right?")

        self.window.blit(aIcon, ((self.width-40), (self.height/2)-50))
        # Left Button Implementation
        laIcon = pygame.transform.rotate(aIcon, 180)

        goLeft = button.Button(
            '', (self.width/4), (self.height/2)-75, 35, 100,
            std_font, off_white, button_selected, button_unselected)
        goLeft.draw(self.window)
        if (self.clicked and goLeft.isInButton(self.last_click_x, self.last_click_y))  or self.key_pressed == pygame.K_LEFT:
            self.mouseOverButton = True
            self.clicked = False
            self.centerScreenx = self.centerScreenx - 50
            #self.window.blit(testIMG, (300, 46))
            print("Panning to Left?")

        self.window.blit(laIcon, ((self.width/4)-10, (self.height/2) - 50))
        # Up Button Implementation
        uaIcon = pygame.transform.rotate(aIcon, 90)

        goUp = button.Button(
            '', (self.width/2)+75, 0, 100, 35,
            std_font, off_white, button_selected, button_unselected)
        goUp.draw(self.window)
        if (self.clicked and goUp.isInButton(self.last_click_x, self.last_click_y))  or self.key_pressed == pygame.K_UP:
            self.mouseOverButton = True
            self.clicked = False
            self.centerScreeny = self.centerScreeny - 50
            print("Panning Up?")

        self.window.blit(uaIcon, ((self.width/2) + 100, -10))
        # Implements Down Button
        daIcon = pygame.transform.rotate(aIcon, 270)
        goDown = button.Button(
            '', ((self.width/2)+75), (self.height-35), 100, 35,
            std_font, off_white, button_selected, button_unselected)
        goDown.draw(self.window)
        if (self.clicked and goDown.isInButton(self.last_click_x, self.last_click_y))  or self.key_pressed == pygame.K_DOWN:
            self.mouseOverButton = True
            self.clicked = False
            self.centerScreeny = self.centerScreeny + 50
            print("Panning Down?")

        self.window.blit(daIcon, ((self.width/2) + 100, (self.height-40)))

        #
        # Creates button and overlays image for settings button
        #

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

            pygame.draw.rect(self.window, (150, 150, 150), pygame.Rect(
                2 * (self.window.get_width() / 3), 0, self.window.get_width(), self.window.get_height()))

        self.window.blit(photo, (self.window.get_width() -
                         60, self.window.get_height()-60))

    def drawDimensionsText(self):
        self.font = pygame.font.SysFont('Arial', 25)
        self.window.blit(self.font.render('Enter puzzle dimensions', (255,
                         255, 255), (255, 255, 255)), ((self.window.get_width()/3)*2, 10))
        self.window.blit(self.font.render('Enter Amount of generations', (255,
                         255, 255), (255, 255, 255)), ((self.window.get_width()/3)*2, 93))
        self.window.blit(self.font.render('Enter Generation Size', (255,
                         255, 255), (255, 255, 255)), ((self.window.get_width()/3)*2, 93*2))

    def drawSettingsButton(self):
        # Draws the settings button logo

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
        # Draws the close 'X' for the settings close button.

        std_font = pygame.font.Font(pygame.font.get_default_font(), 48)
        off_white = (230, 230, 230)
        button_selected = (80, 80, 80)
        button_unselected = (200, 200, 200)

        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        my_file = os.path.join(THIS_FOLDER, 'closeX.png')

        DEFAULT_IMAGE_SIZE = (55, 55)

        closeX = pygame.image.load(my_file)

        closeX = pygame.transform.scale(closeX, DEFAULT_IMAGE_SIZE)

        # If the settings window is open, draw the close button.

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

    def updateImage(self):
        import shutil
        shutil.copy("result2.jpg","result.jpg")
        time.sleep(0.1)
        testIMG = pygame.image.load(self.resultImage)
        testIMG.convert()
        w = testIMG.get_width()
        h = testIMG.get_height()
        if w / h != self.x / self.y:
            max_dim = max(w, h)
            if max_dim == w:
                self.x = max(self.x, self.y)
                self.y = int(self.x * (h / w))
            else:
                self.y = max(self.x, self.y)
                self.x = int(self.y * (w / h))
            self.x_step = self.x // 10
            self.y_step = self.y // 10

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
            self.drawDimensionsText()
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
        prev_result_updated = 0
        while self.running:
            now = time.time_ns()
            delta = delta + ((now - last) / timePerFrame)
            last = now
            
            try:
                result_updated = os.stat("result2.jpg").st_mtime
                if result_updated > prev_result_updated:
                    prev_result_updated = result_updated
                    self.updateImage()
            except:
                pass

            if delta >= 1:
                # Call update and render if on time
                self.render()
                self.update()
                if not self.running:
                    break

                delta = delta - 1
        # Quit pygame when loop completes
        pygame.quit()