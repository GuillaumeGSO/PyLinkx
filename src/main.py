import pygame
import sys
from game import Game

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 30
BOARD_TOP_MARGIN = 80  # Space above the board for scores/info
BOARD_MARGIN = 20      # Margin all around the board
BOARD_WIDTH = 400
BOARD_HEIGHT = 250

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
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
                pygame.draw.rect(screen, BLUE, rect, 1)  # outline


def draw_player_pieces(screen, player, hover_idx=None, hover_x=None, rects=None, scale=0.5):
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
        #TODO draw 2 lines of 7 pieces max
        if hover_idx == idx and hover_x is not None:
            # Don't draw the selected piece in the bottom row
            if rects is not None:
                rects.append(pygame.Rect(hover_x, y, piece_widths[idx], len(piece.shape) * scaled_block))
        else:
            draw_shape(screen, piece.shape, x, y, block_size, color=player.color, scale=scale)
            if rects is not None:
                rects.append(pygame.Rect(x, y, piece_widths[idx], len(piece.shape) * scaled_block))
        x += piece_widths[idx] + 20 # Spacing between pieces


def draw_selected_piece(screen, piece, x, y, block_size, color):
    # Draw the selected piece above the grid, centered horizontally, at scale 1
    board_rect = pygame.Rect(
        (SCREEN_WIDTH - BOARD_WIDTH) // 2,
        BOARD_TOP_MARGIN + (SCREEN_HEIGHT - BOARD_TOP_MARGIN - BOARD_HEIGHT) // 2,
        BOARD_WIDTH,
        BOARD_HEIGHT
    )
    piece_height = len(piece.shape) * block_size
    #Use piece_height to avoid drawing off-screen
    draw_shape(screen, piece.shape, x, y, block_size, color, scale=1.0)


def rotate_shape(shape):
    # Rotate a 2D list (matrix) clockwise
    return [list(row) for row in zip(*shape[::-1])]

def flip_shape(shape):
    # Flip horizontally
    return [list(row[::-1]) for row in shape]


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('PyLinkx Pygame Project')
    clock = pygame.time.Clock()

    game = Game()  # Initialize game logic
    font = pygame.font.SysFont(None, 36)
    turn = 1  # 0 for player 1, 1 for player 2
    selected_piece_idx = 0
    piece_x = 0  # grid-aligned x for the selected piece
    drag_shape = [row[:] for row in game.players[turn].pieces[selected_piece_idx].shape]
    block_size = BOARD_WIDTH // 9
    board_rect = pygame.Rect(
        (SCREEN_WIDTH - BOARD_WIDTH) // 2,
        BOARD_TOP_MARGIN + (SCREEN_HEIGHT - BOARD_TOP_MARGIN - BOARD_HEIGHT) // 2,
        BOARD_WIDTH,
        BOARD_HEIGHT
    )
    piece_x = board_rect.left  # Start at leftmost position
    piece_y = None  # Not dropped yet

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_TAB:
                    # Cycle through available pieces
                    selected_piece_idx = (selected_piece_idx + 1) % len(game.players[turn].pieces)
                    drag_shape = [row[:] for row in game.players[turn].pieces[selected_piece_idx].shape]
                    piece_x = board_rect.left
                    piece_y = None
                elif event.key == pygame.K_RETURN:
                    # Flip the piece horizontally
                    drag_shape = flip_shape(drag_shape)
                elif event.key == pygame.K_LEFT:
                    # Move piece left
                    min_x = board_rect.left
                    if piece_x - block_size >= min_x:
                        piece_x -= block_size
                elif event.key == pygame.K_RIGHT:
                    # Move piece right
                    max_x = board_rect.right - len(drag_shape[0]) * block_size
                    if piece_x + block_size <= max_x:
                        piece_x += block_size
                elif event.key == pygame.K_UP:
                    # Rotate piece
                    new_shape = rotate_shape(drag_shape)
                    # Adjust x if needed to keep within board
                    new_width = len(new_shape[0]) * block_size
                    max_x = board_rect.right - new_width
                    if piece_x > max_x:
                        piece_x = max_x
                    drag_shape = new_shape
                elif event.key == pygame.K_DOWN:
                    # Drop the piece (logic to be implemented)
                    # Need to verify if move is valid (not creating hole under the piece)
                    new_height = len(drag_shape) * block_size
                    piece_y = board_rect.bottom - new_height
                    # release the piece and reset for next player
                    turn = (turn + 1) % len(game.players) #Do that in a function later
                    # When a piece is played, it should be removed from player's shapes
                    #TODO


        game.update()  # Update game logic

        screen.fill(BLACK)
        # Draw area for scores/game info
        pygame.draw.rect(screen, BLACK, (0, 0, SCREEN_WIDTH, BOARD_TOP_MARGIN))
        # Display player names and scores
        for i, player in enumerate(game.players):
            text = font.render(f"{player.name} - Score: {player.score}", True, WHITE)
            screen.blit(text, (30, 10 + i * 36))
        current_player = game.players[turn]
        # Display current player's pieces below the grid at scale 0.5
        draw_player_pieces(screen, current_player, font, scale=0.5)
        # Draw selected piece above the grid at scale 1
        piece_height = len(drag_shape) * block_size
        if piece_y==None:
            y = board_rect.top - piece_height - 10  # 10px above the board
        else:
            y = piece_y
        draw_grid(screen)
        draw_selected_piece(screen, type('Tmp', (), {'shape': drag_shape})(), piece_x, y, block_size, current_player.color)
        # Draw your game objects here, possibly using game state

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
