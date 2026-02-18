from tkinter import font
import pygame

from game import Game
from piece import Piece


class GameRenderer:

    BOARD_TOP_MARGIN = 80  # Space above the board for scores/info
    BOARD_MARGIN = 20  # Margin all around the board
    BOARD_SIZE = 360  # 360 is divisible by 9 (40px per cell)
    BOARD_WIDTH = BOARD_SIZE
    BOARD_HEIGHT = BOARD_SIZE

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 255)
    INFO_BG = (30, 30, 30)
    BOARD_BG = (30, 30, 30)

    """Pure display: knows about fonts, colors, and the screen."""

    def __init__(self, screen, game: Game):
        self.screen = screen
        self.game = game
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = screen.get_size()
        self.font = pygame.font.SysFont("Arial", 28)
        self.replay_rect = pygame.Rect(0, 0, 0, 0)
        self.board_rect = pygame.Rect(
            (self.SCREEN_WIDTH - GameRenderer.BOARD_WIDTH) // 2,
            GameRenderer.BOARD_TOP_MARGIN
            + (
                self.SCREEN_HEIGHT
                - GameRenderer.BOARD_TOP_MARGIN
                - GameRenderer.BOARD_HEIGHT
            )
            // 2,
            GameRenderer.BOARD_WIDTH,
            GameRenderer.BOARD_HEIGHT,
        )
        self.block_size = GameRenderer.BOARD_WIDTH // Game.GRID_SIZE

    def draw(self):
        self.screen.fill(self.BLACK)

        # self.draw_score()
        self.draw_board()
        self.draw_grid()
        if self.game.status == Game.PLAYING:
            self.draw_selected_piece(
                self.game.current_piece, color=self.game.current_piece.color
            )
            self.draw_ghost_piece(self.game.current_piece)
            self.draw_scores()

        elif self.game.status == Game.GAMEOVER:
            
            # Draw Winner
            if self.game.winner:
                msg = self.font.render(f"{self.game.winner.name} Wins !", True, "green")
            elif len(self.game.get_players_in_play()) == 0:
                msg = self.font.render("Winner by zone !", True, "green")
                self.draw_scores()
            else:
                msg = self.font.render("It's a Tie !", True, "green")
            self.screen.blit(
                msg,
                (
                    self.SCREEN_WIDTH // 2 - msg.get_width() // 2,
                    self.SCREEN_HEIGHT // 8,
                ),
            )

            # Draw Replay Button
            btn_txt = self.font.render("[ Press R to replay ]", True, "yellow")
            self.replay_rect = btn_txt.get_rect(
                center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 5)
            )
            self.screen.blit(btn_txt, self.replay_rect)

    def draw_scores(self):
        for i, player in enumerate(self.game.players):
            text = self.font.render(
                f"{player.name}: {player.score}", True, player.color
            )
            self.screen.blit(text, (30, 10 + i * 36))

    def draw_board(self):
        # Center the board
        pygame.draw.rect(self.screen, GameRenderer.BOARD_BG, self.board_rect)
        # Draw columns (visible lines)
        col_width = GameRenderer.BOARD_WIDTH // Game.GRID_SIZE
        for col in range(1, Game.GRID_SIZE):
            x = self.board_rect.left + col * col_width
            pygame.draw.line(
                self.screen,
                GameRenderer.WHITE,
                (x, self.board_rect.top),
                (x, self.board_rect.bottom),
                1,
            )

    def draw_grid(self):
        cell_w = GameRenderer.BOARD_WIDTH // Game.GRID_SIZE
        cell_h = GameRenderer.BOARD_HEIGHT // Game.GRID_SIZE

        for row_idx, row in enumerate(self.game.grid):
            for col_idx, cell_value in enumerate(row):
                if cell_value != 0:
                    rect = pygame.Rect(
                        self.board_rect.left + col_idx * cell_w,
                        self.board_rect.top + row_idx * cell_h,
                        cell_w,
                        cell_h,
                    )
                    color = self.game.players[cell_value - 1].color
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, GameRenderer.BLUE, rect, 1)

    def draw_shape(self, piece: Piece, x_pixel, y_pixel, color=WHITE, scale=1.0):
        scaled_block = int(self.block_size * scale)
        for row_idx, row in enumerate(piece.shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    rect = pygame.Rect(
                        x_pixel + col_idx * scaled_block,
                        y_pixel + row_idx * scaled_block,
                        scaled_block,
                        scaled_block,
                    )
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, GameRenderer.BLUE, rect, 1)  # outline

    def draw_ghost_piece(self, piece: Piece):
        if self.game.ghost_grid_y is None:
            return

        x_pixel = piece.x * self.block_size + self.board_rect.left
        y_pixel = self.board_rect.top + (self.game.ghost_grid_y * self.block_size)  # type: ignore

        self.draw_shape(
            piece,
            x_pixel,
            y_pixel,
            color=(50, 50, 50),
        )

    def draw_player_pieces(
        self, player, hover_idx=None, hover_x=None, rects=None, scale=0.5
    ):
        # Display player's pieces as Tetris shapes below the grid, centered horizontally, at scale 0.5
        board_rect = pygame.Rect(
            (self.SCREEN_WIDTH - GameRenderer.BOARD_WIDTH) // 2,
            GameRenderer.BOARD_TOP_MARGIN
            + (
                self.SCREEN_HEIGHT
                - GameRenderer.BOARD_TOP_MARGIN
                - GameRenderer.BOARD_HEIGHT
            )
            // 2,
            GameRenderer.BOARD_WIDTH,
            GameRenderer.BOARD_HEIGHT,
        )

        scaled_block = int(self.block_size * scale)
        # Calculate total width of all pieces for centering
        piece_widths = []
        for piece in player.pieces:
            shape = piece.shape
            width = len(shape[0]) * scaled_block
            piece_widths.append(width)
        total_width = sum(piece_widths) + (len(piece_widths) - 1) * 20
        x = board_rect.left + (GameRenderer.BOARD_WIDTH - total_width) // 2
        y = board_rect.bottom + 20  # 20px below the board
        if rects is not None:
            rects.clear()
        for idx, piece in enumerate(player.pieces):
            # TODO draw 2 lines of 7 pieces max
            if hover_idx == idx and hover_x is not None:
                # Don't draw the selected piece in the bottom row
                if rects is not None:
                    rects.append(
                        pygame.Rect(
                            hover_x, y, piece_widths[idx], piece.height() * scaled_block
                        )
                    )
            else:
                self.draw_shape(piece, x, y, color=player.color, scale=scale)
                if rects is not None:
                    rects.append(
                        pygame.Rect(
                            x, y, piece_widths[idx], piece.height() * scaled_block
                        )
                    )
            x += piece_widths[idx] + 20  # Spacing between pieces

    def draw_selected_piece(self, piece, color):
        x_pixel = piece.x * self.block_size + self.board_rect.left
        y_pixel = (
            self.board_rect.top - self.block_size * piece.height()
        )  # Start above the board
        # Drawing above the board
        self.draw_shape(piece, x_pixel, y_pixel, color=color, scale=1.0)
