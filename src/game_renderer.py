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
    BOARD_BG = (40, 40, 40)

    """Pure display: knows about fonts, colors, and the screen."""

    def __init__(self, screen, game:Game):
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
        self.piece_x = self.board_rect.left
        self.piece_y = None
        self.ghost_grid_y = -1
        
    def draw(self):
        self.screen.fill((30, 30, 30))

        if self.game.status == Game.PLAYING:
            # txt = self.font.render(f"Playing... ", True, "green")
            # self.screen.blit(txt, (70, 130))
            pass

        elif self.game.status == Game.GAMEOVER:
            # Draw Winner
            # msg = self.font.render(f"{game.winner} Wins!", True, "green")
            # self.screen.blit(msg, (130, 100))

            # # Draw Replay Button
            # btn_txt = self.font.render("[ REPLAY ]", True, "yellow")
            # self.replay_rect = btn_txt.get_rect(center=(200, 200))
            # self.screen.blit(btn_txt, self.replay_rect)
            pass
        # # 1. UI: Scores and Player Pieces
        # for i, player in enumerate(game.players):
        #     text = font.render(
        #         f"{player.name}: {player.score}", True, GameRenderer.WHITE
        #     )
        #     screen.blit(text, (30, 10 + i * 36))
        # renderer.draw_player_pieces(screen, game.players[turn], scale=0.5)

        # # 2. GRID: Draw the static board and placed pieces
        # renderer.draw_grid(screen)
        # renderer.draw_board(screen, game, board_rect)

        # # 3. GHOST PIECE CALCULATION
        # grid_x = (piece_x - board_rect.left) // block_size
        # ghost_grid_y = -1
        
        


    def draw_grid(self, screen):
        # Center the board
        pass
        
        # pygame.draw.rect(screen, GameRenderer.BOARD_BG, board_rect)
        # # Draw 9 columns (visible lines)
        # col_width = GameRenderer.BOARD_WIDTH // 9
        # for col in range(1, 9):
        #     x = board_rect.left + col * col_width
        #     pygame.draw.line(
        #         screen,
        #         GameRenderer.WHITE,
        #         (x, board_rect.top),
        #         (x, board_rect.bottom),
        #         1,
        #     )
        # 9 rows (not visible, so nothing drawn)

    def draw_board(self, screen, game, board_rect):
        # Calculer la taille précise d'une cellule pour qu'elle tienne dans le rectangle
        cell_w = GameRenderer.BOARD_WIDTH // 9
        cell_h = GameRenderer.BOARD_HEIGHT // 9

        for row_idx, row in enumerate(game.grid):
            for col_idx, cell_value in enumerate(row):
                if cell_value != 0:
                    # On utilise cell_w et cell_h pour remplir exactement la case de la grille
                    rect = pygame.Rect(
                        board_rect.left + col_idx * cell_w,
                        board_rect.top + row_idx * cell_h,
                        cell_w,
                        cell_h,
                    )
                    color = game.players[cell_value - 1].color
                    pygame.draw.rect(screen, color, rect)
                    pygame.draw.rect(screen, GameRenderer.BLUE, rect, 1)

    def draw_shape(
        self, screen, piece: Piece, x, y, block_size, color=WHITE, scale=1.0
    ):
        scaled_block = int(block_size * scale)
        for row_idx, row in enumerate(piece.shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    rect = pygame.Rect(
                        x + col_idx * scaled_block,
                        y + row_idx * scaled_block,
                        scaled_block,
                        scaled_block,
                    )
                    pygame.draw.rect(screen, color, rect)
                    pygame.draw.rect(screen, GameRenderer.BLUE, rect, 1)  # outline

    def draw_player_pieces(
        self, screen, player, hover_idx=None, hover_x=None, rects=None, scale=0.5
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
        block_size = GameRenderer.BOARD_WIDTH // 9
        scaled_block = int(block_size * scale)
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
                self.draw_shape(
                    screen, piece, x, y, block_size, color=player.color, scale=scale
                )
                if rects is not None:
                    rects.append(
                        pygame.Rect(
                            x, y, piece_widths[idx], piece.height() * scaled_block
                        )
                    )
            x += piece_widths[idx] + 20  # Spacing between pieces

    def draw_selected_piece(self, screen, piece, x, y, block_size, color):
        self.draw_shape(screen, piece, x, y, block_size, color=color, scale=1.0)
