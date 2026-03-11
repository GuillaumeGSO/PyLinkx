import pygame
import sys
from game import Game, Actions
from game_renderer import GameRenderer

# Constants
FPS = 10


def main():
    pygame.init()
    screen = pygame.display.set_mode(
        (GameRenderer.SCREEN_WIDTH, GameRenderer.SCREEN_HEIGHT)
    )
    pygame.display.set_caption("PyLinkx Pygame Project")
    clock = pygame.time.Clock()

    game = Game()  # Initialize game logic
    renderer = GameRenderer(screen, game)
    game.start_turn()
    running = True

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if game.status == game.PLAYING:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_TAB:
                        game.execute_action(Actions.ACTION_CYCLE_PIECE)
                    elif event.key == pygame.K_LEFT:
                        game.execute_action(Actions.ACTION_MOVE_LEFT)
                    elif event.key == pygame.K_RIGHT:
                        game.execute_action(Actions.ACTION_MOVE_RIGHT)
                    elif event.key == pygame.K_UP:
                        game.execute_action(Actions.ACTION_ROTATE)
                    elif event.key == pygame.K_RETURN:
                        game.execute_action(Actions.ACTION_FLIP)
                    elif event.key == pygame.K_DOWN:
                        game.execute_action(Actions.ACTION_DROP)
                    elif event.key == pygame.K_p:
                        # Give up and switch turn
                        game.give_up_and_check(game.current_player)
                        game.current_player = game.get_next_player()
                        game.start_turn()
                        game.update()
                elif game.status == game.GAMEOVER:
                    # should render button to reset game
                    print("Game Over! Press R to Restart or ESC to Quit.")
                    if event.key == pygame.K_r:
                        game.reset()
                        game.start_turn()
                    elif event.key == pygame.K_ESCAPE:
                        running = False
        renderer.draw()
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
