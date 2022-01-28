import pygame


class Button:
    def __init__(self, text: str, x: int, y: int, width: int, height: int, font: pygame.font.Font, color: tuple[3], color_selected: tuple[3], color_unselected: tuple[3]):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = font.render(text, True, color)
        self.color_s = color_selected
        self.color_u = color_unselected

    def isInButton(self, x: int, y: int):
        if x > self.x:
            if x < self.x + self.width:
                if y > self.y:
                    if y < self.y + self.height:
                        return True
        return False

    def draw(self, window):
        mouse = pygame.mouse.get_pos()
        if self.isInButton(mouse[0], mouse[1]):
            pygame.draw.rect(window, self.color_s, pygame.Rect(
                self.x, self.y, self.width, self.height))
        else:
            pygame.draw.rect(window, self.color_u, pygame.Rect(
                self.x, self.y, self.width, self.height))
        button_rect = self.text.get_rect()
        button_rect.center = (self.x + (self.width / 2),
                              self.y + (self.height / 2))
        window.blit(self.text, button_rect)
