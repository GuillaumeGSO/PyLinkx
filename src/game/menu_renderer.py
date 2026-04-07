import sys

import pygame

from .game_renderer import _draw_gradient, _FONT

_IS_WASM = sys.platform == "emscripten"


class MenuRenderer:

    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 600

    MENU_ITEMS = (["2 Human Players", "How to Play"] if _IS_WASM
                  else ["2 Human Players", "Human vs Computer", "How to Play"])
    DIFF_ITEMS = ["Easy", "Medium", "Hard", "< Back"]

    GOLD = (255, 215, 0)
    WHITE = (255, 255, 255)
    DIM = (160, 150, 180)
    ACTIVE_BORDER = (140, 80, 200)

    HOW_TO_PLAY_LINES = [
        ("CONTROLS", True),
        ("  TAB          Cycle piece", False),
        ("  LEFT / RIGHT  Move piece", False),
        ("  UP           Rotate", False),
        ("  ENTER        Flip", False),
        ("  DOWN         Drop (place piece)", False),
        ("  P            Pass turn  (if no valid moves)", False),
        ("  ESC          Return to menu", False),
        ("", False),
        ("WIN CONDITIONS", True),
        ("  Path win:  connect opposite edges with your color", False),
        ("  Score win: largest connected group when all pieces placed", False),
        ("             (path win takes priority)", False),
        ("", False),
        ("Press any key to return", False),
    ]

    def __init__(self, screen):
        self.screen = screen
        self.font_title = pygame.font.Font(_FONT, 40)
        self.font_subtitle = pygame.font.Font(_FONT, 12)
        self.font_item = pygame.font.Font(_FONT, 18)
        self.font_hint = pygame.font.Font(_FONT, 10)
        self.font_section = pygame.font.Font(_FONT, 13)

    # ------------------------------------------------------------------
    # Shared
    # ------------------------------------------------------------------

    def draw_gradient_background(self):
        _draw_gradient(
            self.screen,
            (15, 0, 30),
            (0, 0, 0),
            pygame.Rect(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT),
        )

    def _draw_title(self):
        title = self.font_title.render("LINKX", True, self.GOLD)
        self.screen.blit(title, (self.SCREEN_WIDTH // 2 - title.get_width() // 2, 60))

    def _draw_items(self, items, cursor, start_y):
        cx = self.SCREEN_WIDTH // 2
        y = start_y
        for i, label in enumerate(items):
            selected = (i == cursor)
            color = self.GOLD if selected else self.DIM
            surf = self.font_item.render(label, True, color)
            x = cx - surf.get_width() // 2
            self.screen.blit(surf, (x, y))
            if selected:
                arrow = self.font_item.render(">", True, self.ACTIVE_BORDER)
                self.screen.blit(arrow, (x - arrow.get_width() - 10, y))
            y += surf.get_height() + 12

    # ------------------------------------------------------------------
    # Main menu
    # ------------------------------------------------------------------

    def draw_menu(self, cursor: int):
        self.draw_gradient_background()
        self._draw_title()
        self._draw_items(self.MENU_ITEMS, cursor, start_y=200)
        hint = self.font_hint.render("UP/DN Navigate   ENTER Select", True, self.DIM)
        self.screen.blit(hint, (self.SCREEN_WIDTH // 2 - hint.get_width() // 2,
                                self.SCREEN_HEIGHT - 40))

    # ------------------------------------------------------------------
    # Difficulty sub-menu
    # ------------------------------------------------------------------

    def draw_difficulty(self, cursor: int):
        self.draw_gradient_background()
        self._draw_title()
        sub = self.font_subtitle.render("SELECT DIFFICULTY", True, self.DIM)
        self.screen.blit(sub, (self.SCREEN_WIDTH // 2 - sub.get_width() // 2, 185))
        self._draw_items(self.DIFF_ITEMS, cursor, start_y=215)

        note = self.font_hint.render("You will play as  PLAYER 2", True, self.DIM)
        self.screen.blit(note, (self.SCREEN_WIDTH // 2 - note.get_width() // 2,
                                self.SCREEN_HEIGHT - 60))
        hint = self.font_hint.render("UP/DN Navigate   ENTER Select   ESC Back", True, self.DIM)
        self.screen.blit(hint, (self.SCREEN_WIDTH // 2 - hint.get_width() // 2,
                                self.SCREEN_HEIGHT - 36))

    # ------------------------------------------------------------------
    # How to Play
    # ------------------------------------------------------------------

    def draw_how_to_play(self):
        self.draw_gradient_background()

        header = self.font_item.render("HOW TO PLAY", True, self.GOLD)
        self.screen.blit(header, (self.SCREEN_WIDTH // 2 - header.get_width() // 2, 40))

        pygame.draw.line(self.screen, self.ACTIVE_BORDER,
                         (80, 100), (self.SCREEN_WIDTH - 80, 100), 1)

        y = 118
        left = 120
        line_h_normal = self.font_hint.get_height() + 4
        line_h_section = self.font_section.get_height() + 6

        for text, is_section in self.HOW_TO_PLAY_LINES:
            if not text:
                y += 10
                continue
            if is_section:
                surf = self.font_section.render(text, True, self.GOLD)
                self.screen.blit(surf, (left, y))
                y += line_h_section
            else:
                surf = self.font_hint.render(text, True, self.WHITE)
                self.screen.blit(surf, (left, y))
                y += line_h_normal

    # ------------------------------------------------------------------
    # Loading screen
    # ------------------------------------------------------------------

    def draw_loading(self, label: str):
        self.draw_gradient_background()
        self._draw_title()
        msg = self.font_item.render(f"Loading {label}...", True, self.DIM)
        self.screen.blit(msg, (self.SCREEN_WIDTH // 2 - msg.get_width() // 2,
                               self.SCREEN_HEIGHT // 2 - msg.get_height() // 2))
