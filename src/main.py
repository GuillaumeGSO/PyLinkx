import pygame
import sys
from game import Game

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
BOARD_TOP_MARGIN = 80  # Space above the board for scores/info
BOARD_MARGIN = 20      # Margin all around the board
BOARD_WIDTH = 400
BOARD_HEIGHT = 250

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
INFO_BG = (30, 30, 30)
BOARD_BG = (40, 40, 40)


def draw_grid(screen):
    # Center the board
    board_rect = pygame.Rect(
        (SCREEN_WIDTH - BOARD_WIDTH) // 2,
        BOARD_TOP_MARGIN + (SCREEN_HEIGHT - BOARD_TOP_MARGIN - BOARD_HEIGHT) // 2,
        BOARD_WIDTH,
        BOARD_HEIGHT
    )
    pygame.draw.rect(screen, BOARD_BG, board_rect)
    # Draw 9 columns (visible lines)
    col_width = BOARD_WIDTH // 9
    for col in range(1, 9):
        x = board_rect.left + col * col_width
        pygame.draw.line(screen, WHITE, (x, board_rect.top), (x, board_rect.bottom), 2)
    # 9 rows (not visible, so nothing drawn)


def draw_shape(screen, shape, x, y, block_size, color=WHITE, scale=1.0):
    scaled_block = int(block_size * scale)
    for row_idx, row in enumerate(shape):
        for col_idx, cell in enumerate(row):
            if cell:
                rect = pygame.Rect(
                    x + col_idx * scaled_block,
                    y + row_idx * scaled_block,
                    scaled_block,
                    scaled_block
                )
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLACK, rect, 2)  # outline


def draw_player_pieces(screen, player, font, hover_idx=None, hover_x=None, rects=None, scale=0.5):
    # Display player's pieces as Tetris shapes below the grid, centered horizontally, at scale 0.5
    board_rect = pygame.Rect(
        (SCREEN_WIDTH - BOARD_WIDTH) // 2,
        BOARD_TOP_MARGIN + (SCREEN_HEIGHT - BOARD_TOP_MARGIN - BOARD_HEIGHT) // 2,
        BOARD_WIDTH,
        BOARD_HEIGHT
    )
    block_size = BOARD_WIDTH // 9
    scaled_block = int(block_size * scale)
    # Calculate total width of all pieces for centering
    piece_widths = []
    for piece in player.pieces:
        shape = piece.shape
        width = len(shape[0]) * scaled_block
        piece_widths.append(width)
    total_width = sum(piece_widths) + (len(piece_widths) - 1) * 20
    x = board_rect.left + (BOARD_WIDTH - total_width) // 2
    y = board_rect.bottom + 20  # 20px below the board
    if rects is not None:
        rects.clear()
    for idx, piece in enumerate(player.pieces):
        if hover_idx == idx and hover_x is not None:
            # Don't draw the selected piece in the bottom row
            if rects is not None:
                rects.append(pygame.Rect(hover_x, y, piece_widths[idx], len(piece.shape) * scaled_block))
        else:
            draw_shape(screen, piece.shape, x, y, block_size, color=player.color, scale=scale)
            if rects is not None:
                rects.append(pygame.Rect(x, y, piece_widths[idx], len(piece.shape) * scaled_block))
        x += piece_widths[idx] + 20


def draw_selected_piece(screen, piece, x, block_size):
    # Draw the selected piece above the grid, centered horizontally, at scale 1
    board_rect = pygame.Rect(
        (SCREEN_WIDTH - BOARD_WIDTH) // 2,
        BOARD_TOP_MARGIN + (SCREEN_HEIGHT - BOARD_TOP_MARGIN - BOARD_HEIGHT) // 2,
        BOARD_WIDTH,
        BOARD_HEIGHT
    )
    piece_height = len(piece.shape) * block_size
    y = board_rect.top - piece_height - 10  # 10px above the board
    draw_shape(screen, piece.shape, x, y, block_size, color=(255,255,255), scale=1.0)


def rotate_shape(shape):
    # Rotate a 2D list (matrix) clockwise
    return [list(row) for row in zip(*shape[::-1])]


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('PyLinkx Pygame Project')
    clock = pygame.time.Clock()

    game = Game()  # Initialize game logic
    font = pygame.font.SysFont(None, 36)
    turn = 0  # 0 for player 1, 1 for player 2
    dragging = False
    drag_piece_idx = None
    drag_offset_x = 0
    drag_x = 0
    drag_rot = 0  # Track rotation state for the dragged piece
    piece_rects = []
    drag_shape = None  # Store the current shape being dragged

    running = True
    while running:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not dragging:
                draw_player_pieces(screen, game.players[turn], font, rects=piece_rects)
                for idx, rect in enumerate(piece_rects):
                    if rect.collidepoint(mouse_x, mouse_y):
                        dragging = True
                        drag_piece_idx = idx
                        drag_offset_x = mouse_x - rect.x
                        drag_rot = 0
                        # Copy the shape so we can rotate independently
                        drag_shape = [row[:] for row in game.players[turn].pieces[idx].shape]
                        break
            elif event.type == pygame.MOUSEBUTTONUP and dragging:
                dragging = False
                drag_piece_idx = None
                drag_shape = None
            elif event.type == pygame.KEYDOWN and dragging and drag_piece_idx is not None:
                if event.key == pygame.K_SPACE:
                    # Rotate the piece (work on drag_shape, not the player's piece)
                    drag_shape = rotate_shape(drag_shape)
                    drag_rot = (drag_rot + 1) % 4

        game.update()  # Update game logic

        screen.fill(BLACK)
        # Draw area for scores/game info
        pygame.draw.rect(screen, BLACK, (0, 0, SCREEN_WIDTH, BOARD_TOP_MARGIN))
        # Display player names and scores
        for i, player in enumerate(game.players):
            text = font.render(f"{player.name} - Score: {player.score}", True, WHITE)
            screen.blit(text, (30, 10 + i * 36))
        # Display current player's pieces below the grid at scale 0.5
        draw_player_pieces(screen, game.players[turn], font, rects=piece_rects, scale=0.5)
        # Draw selected piece above the grid at scale 1 if dragging
        if dragging and drag_piece_idx is not None and drag_shape is not None:
            board_rect = pygame.Rect(
                (SCREEN_WIDTH - BOARD_WIDTH) // 2,
                BOARD_TOP_MARGIN + (SCREEN_HEIGHT - BOARD_TOP_MARGIN - BOARD_HEIGHT) // 2,
                BOARD_WIDTH,
                BOARD_HEIGHT
            )
            block_size = BOARD_WIDTH // 9
            piece_width = len(drag_shape[0]) * block_size
            min_x = board_rect.left
            max_x = board_rect.right - piece_width
            snapped_x = min(max(min_x, mouse_x - drag_offset_x), max_x)
            snapped_x = min_x + ((snapped_x - min_x) // block_size) * block_size
            draw_selected_piece(screen, type('Tmp', (), {'shape': drag_shape})(), snapped_x, block_size)
        draw_grid(screen)
        # Draw your game objects here, possibly using game state

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
