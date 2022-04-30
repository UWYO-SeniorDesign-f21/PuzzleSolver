import pygame
vec = pygame.math.Vector2


def ignore(key):
    if key == pygame.K_RETURN:
        return True
    if key == pygame.K_KP_ENTER:
        return True
    if key == pygame.K_NUMLOCK:
        return True
    if key == pygame.K_LSHIFT:
        return True
    if key == pygame.K_RSHIFT:
        return True
    if key == pygame.K_TAB:
        return True
    if key == pygame.K_CAPSLOCK:
        return True
    if key == pygame.K_LCTRL:
        return True
    if key == pygame.K_RCTRL:
        return True
    if key == pygame.K_LALT:
        return True
    if key == pygame.K_RALT:
        return True
    if key == pygame.K_KP_PERIOD:
        return True
    if key == pygame.K_KP_DIVIDE:
        return True
    if key == pygame.K_KP_MINUS:
        return True
    if key == pygame.K_KP_PLUS:
        return True
    if key == pygame.K_KP_MULTIPLY:
        return True
    return False


class Text_box:
    def __init__(self, x, y, width, height, bg_colour=(124, 124, 124),
                 active_colour=(255, 255, 255), text_size=18,
                 text_colour=(0, 0, 0), border=0, border_colour=(0, 0, 0)):
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
        self.text_size = text_size
        self.font = pygame.font.SysFont("Times New Roman", self.text_size)
        self.text_colour = text_colour
        self.border = border
        self.border_colour = border_colour

    def update(self):
        pass

    def drawTextBox(self, window):
        if not self.active:
            if self.border == 0:
                self.image.fill(self.bg_colour)
            else:
                self.image.fill(self.border_colour)
                pygame.draw.rect(self.image, self.bg_colour,
                                 (self.border, self.border,
                                  self.width-self.border*2, self.height - self.border*2))

            text = self.font.render(self.text, False, self.text_colour)
            text_height = text.get_height()
            text_width = text.get_width()
            self.image.blit(text, (((self.width - text_width)/2),
                            ((self.height - text_height)/2)))
        else:
            if self.border == 0:
                self.image.fill(self.active_colour)
            else:
                self.image.fill(self.border_colour)
                pygame.draw.rect(self.image, self.active_colour,
                                 (self.border, self.border,
                                  self.width-self.border*2, self.height - self.border*2))
            text = self.font.render(self.text, False, self.text_colour)
            text_height = text.get_height()
            text_width = text.get_width()
            self.image.blit(text, (((self.width - text_width)/2),
                            ((self.height - text_height)/2)))

        window.blit(self.image, self.pos)

    def add_text(self, key):
        if key == pygame.K_BACKSPACE:
            if self.text == "":
                return
            self.text = self.text.rstrip(self.text[-1])
            return
        if ignore(key):
            return
        if key == pygame.K_KP0:
            text = list(self.text)
            text.append('0')
            self.text = "".join(text)
            return
        if key == pygame.K_KP1:
            text = list(self.text)
            text.append('1')
            self.text = "".join(text)
            return
        if key == pygame.K_KP2:
            text = list(self.text)
            text.append('2')
            self.text = "".join(text)
            return
        if key == pygame.K_KP3:
            text = list(self.text)
            text.append('3')
            self.text = "".join(text)
            return
        if key == pygame.K_KP4:
            text = list(self.text)
            text.append('4')
            self.text = "".join(text)
            return
        if key == pygame.K_KP5:
            text = list(self.text)
            text.append('5')
            self.text = "".join(text)
            return
        if key == pygame.K_KP6:
            text = list(self.text)
            text.append('6')
            self.text = "".join(text)
            return
        if key == pygame.K_KP7:
            text = list(self.text)
            text.append('7')
            self.text = "".join(text)
            return
        if key == pygame.K_KP8:
            text = list(self.text)
            text.append('8')
            self.text = "".join(text)
            return
        if key == pygame.K_KP9:
            text = list(self.text)
            text.append('9')
            self.text = "".join(text)
            return
        if not ("" + chr(key)).isalnum():
            return
        text = list(self.text)
        text.append(chr(key))
        self.text = "".join(text)
        # print(self.text)

    def checkTextClick(self, pos):
        if pos[0] > self.x and pos[0] < self.x + self.width:
            if pos[1] > self.y and pos[1] < self.y + self.height:
                self.active = True
            else:
                self.active = False
        else:
            self.active = False
