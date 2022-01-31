import pygame
import button


class FileBox:
    def __init__(self, text: str, x: int, y: int, width: int, height: int, f: pygame.font.Font):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text_render = f.render(
            self.text, True, (230, 230, 230), (50, 50, 50))
        self.text_rect = self.text_render.get_rect()
        self.text_rect.topleft = (x + 10, y + 10)

        rm_box_dim = height - 10
        rm_box_x = x + (width - (rm_box_dim + 5))
        rm_box_y = y + 5
        rm_box_color = (0, 0, 0)
        rm_box_s = (210, 0, 0)
        rm_box_u = (190, 0, 0)
        self.rm_box = button.Button(
            'x', rm_box_x, rm_box_y, rm_box_dim, rm_box_dim, f, rm_box_color, rm_box_s, rm_box_u)

    def isInButton(self, x: int, y: int):
        return self.rm_box.isInButton(x, y)

    def draw(self, window):
        pygame.draw.rect(window, (50, 50, 50), pygame.Rect(
            self.x, self.y, self.width, self.height))
        window.blit(self.text_render, self.text_rect)
        self.rm_box.draw(window)
