from src.player import Player


def test_calculate_score_one_area_aligned():
    player = Player("Alice", (255, 0, 0))
    grid = [[0, 1], [0, 1]]
    assert player.calculate_score(grid, 1) == 2

def test_calculate_score_one_area_not_aligned():
    player = Player("Alice", (255, 0, 0))
    grid = [[1, 0], [0, 1]]
    assert player.calculate_score(grid, 1) == 2

def test_calculate_score_one_area_full_grid():
    player = Player("Alice", (255, 0, 0))
    grid = [[1, 1], [1, 1]]
    assert player.calculate_score(grid, 1) == 4

def test_calculate_score_one_complex_area():
    player = Player("Alice", (255, 0, 0))
    grid = [[1, 1, 1, 0], 
            [1, 1, 0, 0], 
            [0, 0, 1, 1]]
    assert player.calculate_score(grid, 1) == 7
    assert player.calculate_score(grid, 2) == 0
    
def test_calculate_score_two_areas():
    player = Player("Alice", (255, 0, 0))
    grid = [[1, 0, 0, 0], 
            [1, 1, 0, 1], 
            [0, 0, 0, 1]]
    assert player.calculate_score(grid, 1) == 3
    assert player.calculate_score(grid, 2) == 0
    
def test_calculate_score_two_areas_two_players():
    player = Player("Alice", (255, 0, 0))
    grid = [[1, 2, 2, 2], 
            [1, 1, 2, 1], 
            [2, 0, 0, 1]]
    assert player.calculate_score(grid, 1) == 3
    assert player.calculate_score(grid, 2) == 4