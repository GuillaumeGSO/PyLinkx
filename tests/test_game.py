from src.game import Game


def test_L_shape_floating_left_is_invalid():
    game= Game()
    shape = [
        [1, 1],
        [0, 1]
    ]
    assert not game.is_valid_move(shape, 0, 0)


def test_L_shape_supported_is_valid():
    game= Game()
    shape = [
        [1, 0],
        [1, 1]
    ]
    assert game.is_valid_move(shape, 0, 0)


def test_T_shape_is_valid_on_empty_grid():
    game= Game()
    shape = [
        [0, 1, 0],
        [1, 1, 1]
    ]
    assert game.is_valid_move(shape, 0, 0)


def test_L_shape_allowed_only_on_rightmost_column():
    game = Game()
    game.grid[8] = [1,1,1,1,1,1,1,1,0]

    shape = [
        [1, 1],
        [0, 1]
    ]
    for x in range(0, 7):
        assert not game.is_valid_move(shape, x, 7)

    assert game.is_valid_move(shape, 7, 7)


def test_overlap_is_invalid():
    game = Game()
    game.grid[7][7] = 1

    shape = [
        [1]
    ]
    assert not game.is_valid_move(shape, 7, 7)


def test_out_of_bounds_is_invalid():    
    game = Game()
    shape = [
        [1, 1]
    ]
    assert not game.is_valid_move(shape, 8, 0)  # out of right bounds
    assert not game.is_valid_move(shape, 0, 9)  # out of bottom bounds