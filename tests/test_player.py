from src.player import Player


def test_calculate_score_one_area_aligned():
    player = Player("one", 1, None)
    grid = [[0, 1], [0, 1]]
    assert player.calculate_score(grid) == 2


def test_calculate_score_one_area_not_aligned():
    player = Player("one", 1, None)
    grid = [[1, 0], [0, 1]]
    assert player.calculate_score(grid) == 2


def test_calculate_score_one_area_full_grid():
    player = Player("one", 1, None)
    grid = [[1, 1], [1, 1]]
    assert player.calculate_score(grid) == 4


def test_calculate_score_one_complex_area():
    player = Player("one", 1, None)
    grid = [[1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]]
    assert player.calculate_score(grid) == 7


def test_calculate_score_two_areas():
    player1 = Player("one", 1, None)
    player2 = Player("two", 2, None)
    grid = [[1, 0, 0, 0], [1, 1, 0, 1], [0, 0, 0, 1]]
    assert player1.calculate_score(grid) == 3
    assert player2.calculate_score(grid) == 0


def test_calculate_score_two_areas_two_players():
    player1 = Player("one", 1, None)
    player2 = Player("two", 2, None)
    grid = [[1, 2, 2, 2], [1, 1, 2, 1], [2, 0, 0, 1]]
    assert player1.calculate_score(grid) == 3
    assert player2.calculate_score(grid) == 4
