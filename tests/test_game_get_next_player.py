from src.game import Game
from src.player import Player


def test_get_next_player():
    game = Game()
    player1 = Player("one", None, 1)
    player2 = Player("two", None, 2)
    player3 = Player("three", None, 3)
    
    game.players = [player1, player2, player3]
    game.current_player = player1
    assert game.get_next_player() == player2
    game.current_player = player2
    assert game.get_next_player() == player3
    game.current_player = player3
    assert game.get_next_player() == player1
    

def test_player1_give_up():
    game = Game()
    player1 = Player("one", None, 1)
    player2 = Player("two", None, 2)
    player3 = Player("three", None, 3)
    game.players = [player1, player2, player3]
    game.current_player = player1
    player1.give_up()
    assert game.get_next_player() == player2
    
def test_player2_give_up():
    game = Game()
    player1 = Player("one", None, 1)
    player2 = Player("two", None, 2)
    player3 = Player("three", None, 3)
    game.players = [player1, player2, player3]
    game.current_player = player2
    player2.give_up()
    assert game.get_next_player() == player3
    
def test_player3_give_up():
    game = Game()
    player1 = Player("one", None, 1)
    player2 = Player("two", None, 2)
    player3 = Player("three", None, 3)
    game.players = [player1, player2, player3]
    game.current_player = player3
    player3.give_up()
    assert game.get_next_player() == player1

def test_all_players_give_up():
    game = Game()
    player1 = Player("one", None, 1)
    player2 = Player("two", None, 2)
    player3 = Player("three", None, 3)
    game.players = [player1, player2, player3]
    game.current_player = player2
    player1.give_up()
    player2.give_up()
    player3.give_up()
    assert game.get_next_player() == player2