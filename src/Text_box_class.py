import pygame
vec = pygame.math.Vector2


class Text_box:
    def __init__(self, x, y, width, height, bg_colour = (124,124,124), active_colour = (255,255,255)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.pos = vec(x,y)
        self.size = vec(width, height)
        self.image = pygame.Surface((width,height))
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

    def add_text(self,key):
        print(key)

    def checkTextClick(self, pos):
        if pos[0] > self.x and pos[0] < self.width:
            if pos[1] > self.y and pos[1] < self.height: 
               self.active = True
            else:
                self.active = False
        else:
            self.active = False