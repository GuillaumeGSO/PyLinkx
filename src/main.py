import asyncio
import pygame
from game import Game, Actions
from game_renderer import GameRenderer

# Constants
FPS = 10


async def main():
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
    gameover_since = None  # ms timestamp when GAMEOVER first detected

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
                elif game.status == game.GAMEOVER:
                    elapsed = pygame.time.get_ticks() - gameover_since if gameover_since else 0
                    if elapsed >= 3000:
                        if event.key == pygame.K_r:
                            game.reset()
                            game.start_turn()
                            gameover_since = None
                        elif event.key == pygame.K_ESCAPE:
                            running = False

        if game.status == game.GAMEOVER and gameover_since is None:
            gameover_since = pygame.time.get_ticks()

        renderer.draw()
        pygame.display.flip()
        await asyncio.sleep(1.0 / FPS)


if __name__ == "__main__":
    asyncio.run(main())
