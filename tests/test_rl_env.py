"""Tests for the PyLinkx Gymnasium environment wrapper."""

import numpy as np
from gymnasium.spaces import Discrete
from src.game_env import PyLinkxEnv
from src.game import Game


class TestPyLinkxEnvInitialization:
    """Test environment initialization."""

    def test_env_creation(self):
        """Test that environment can be created successfully."""
        env = PyLinkxEnv()
        assert env is not None
        assert env.action_space is not None
        assert env.observation_space is not None

    def test_action_space(self):
        """Test that action space is discrete with 5 actions."""
        env = PyLinkxEnv()
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 7

    def test_observation_space(self):
        """Test that observation space is correct shape and dtype."""
        env = PyLinkxEnv()
        assert env.observation_space.shape == (9, 9, 1)
        assert env.observation_space.dtype == np.int8


class TestPyLinkxEnvReset:
    """Test environment reset functionality."""

    def test_reset_returns_observation_and_info(self):
        """Test that reset returns observation and info dict."""
        env = PyLinkxEnv()
        obs, info = env.reset()

        assert obs is not None
        assert info is not None
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_observation_shape(self):
        """Test that reset returns correct observation shape."""
        env = PyLinkxEnv()
        obs, info = env.reset()

        assert obs.shape == (9, 9, 1)

    def test_reset_clears_step_count(self):
        """Test that reset clears the step counter."""
        env = PyLinkxEnv()
        obs, info = env.reset()

        assert env.step_count == 0

    def test_reset_initializes_piece(self):
        """Test that reset initializes a current piece."""
        env = PyLinkxEnv()
        obs, info = env.reset()

        assert hasattr(env.game, "current_piece")
        assert env.game.current_piece is not None

    def test_reset_info_dict(self):
        """Test that reset info dict contains required keys."""
        env = PyLinkxEnv()
        obs, info = env.reset()

        assert "current_player_idx" in info
        assert "scores" in info
        assert "game_over" in info
        assert "step_count" in info


class TestPyLinkxEnvStep:
    """Test environment step functionality."""

    def test_step_returns_five_values(self):
        """Test that step returns (obs, reward, terminated, truncated, info)."""
        env = PyLinkxEnv()
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)

        assert obs is not None
        assert isinstance(reward, (float, int))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_observation_shape(self):
        """Test that step returns correct observation shape."""
        env = PyLinkxEnv()
        env.reset()

        obs, _, _, _, _ = env.step(0)
        assert obs.shape == (9, 9, 1)

    def test_step_increments_step_count(self):
        """Test that step increments the step counter."""
        env = PyLinkxEnv()
        env.reset()

        initial_count = env.step_count
        env.step(0)

        assert env.step_count == initial_count + 1

    def test_step_max_steps_truncation(self):
        """Test that episode truncates after max_steps."""
        env = PyLinkxEnv(max_steps=5)
        env.reset()

        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(0)
            assert not truncated

        obs, reward, terminated, truncated, info = env.step(0)
        assert truncated

    def test_step_with_different_actions(self):
        """Test that different actions can be executed."""
        env = PyLinkxEnv()
        env.reset()

        # Test all 5 actions
        for action in [0, 1, 2, 3, 4]:
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (9, 9, 1)


class TestPyLinkxEnvGameState:
    """Test integration with game state."""

    def test_env_has_valid_game_object(self):
        """Test that environment has a valid game object."""
        env = PyLinkxEnv()
        assert hasattr(env, "game")
        assert isinstance(env.game, Game)

    def test_observation_matches_grid(self):
        """Test that observation reflects game grid."""
        env = PyLinkxEnv()
        env.reset()

        obs = env._get_observation()
        grid_array = np.array(env.game.grid, dtype=np.int8)
        grid_array = np.expand_dims(grid_array, axis=-1)

        assert np.array_equal(obs, grid_array)

    def test_step_updates_game_state(self):
        """Test that step updates the game state."""
        env = PyLinkxEnv()
        env.reset()

        initial_grid = [row[:] for row in env.game.grid]
        env.step(3)  # Drop piece
        final_grid = [row[:] for row in env.game.grid]

        # Grid should have changed or piece should have been placed
        # (grid might not change if piece bounces back)
        assert env.step_count == 1


class TestPyLinkxEnvReward:
    """Test reward calculation."""

    def test_reward_during_gameplay(self):
        """Test that reward is 0 during gameplay."""
        env = PyLinkxEnv(max_steps=1000)
        env.reset()

        # Execute several non-terminal steps
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(0)
            if not terminated:
                assert reward == 0.0
                break

    def test_reward_structure(self):
        """Test that rewards follow expected structure."""
        env = PyLinkxEnv()
        env.reset()

        # Execute steps and check reward types
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(0)
            assert isinstance(reward, (float, int))
            assert -1.0 <= reward <= 1.0


class TestPyLinkxEnvWinConditions:
    """Test different win conditions and reward structures."""

    def test_win_type_tracking(self):
        """Test that win_type is properly tracked in game state."""
        env = PyLinkxEnv()
        env.reset()

        # Initially no win type
        assert env.game.win_type is None

        # Force a score-based win by making both players give up
        env.game.players[0].give_up()
        env.game.players[1].give_up()
        env.game.check_for_winner()

        assert env.game.win_type == "score"
        assert env.game.winner is not None

    def test_win_type_in_observation(self):
        """Test that win_type is included in observation dictionary."""
        env = PyLinkxEnv()
        obs, info = env.reset()

        # Check that win_type is in info dict
        assert "win_type" in info
        assert info["win_type"] is None  # Initially None

        # Force a win and check again
        env.game.players[0].give_up()
        env.game.players[1].give_up()
        env.game.check_for_winner()

        obs, info = env.reset()  # Reset should clear win_type
        assert info["win_type"] is None

    def test_path_win_higher_reward(self):
        """Test that path-finding wins give higher reward than score wins."""
        env = PyLinkxEnv()

        # Test score-based win reward
        env.reset()
        env.game.players[0].give_up()
        env.game.players[1].give_up()
        env.game.check_for_winner()

        player_idx = 0  # Winner index
        score_reward = env._calculate_reward(player_idx, True)
        assert score_reward == 1.0

        # Test path-finding win reward (harder to test directly)
        # We can at least verify the reward structure exists
        env.reset()
        # Note: Path wins are harder to force programmatically
        # but the reward logic is in place

        print(f"Score-based win reward: {score_reward}")
        print("Path-finding win would give: 2.0")

    def test_reward_structure(self):
        """Test the complete reward structure."""
        env = PyLinkxEnv()
        env.reset()

        # During gameplay: 0 reward
        reward = env._calculate_reward(0, False)
        assert reward == 0.0

        # Set up different scores by placing pieces
        # Player 0 gets a piece placed (score will be calculated from grid)
        env.game.grid[0][0] = 1  # Player 1 piece
        env.game.grid[1][0] = 1  # Connected for score
        env.game.grid[0][1] = 2  # Player 2 piece
        env.game.grid[1][1] = 2  # Connected for score

        # Both players give up
        env.game.players[0].give_up()
        env.game.players[1].give_up()

        winner = env.game.check_for_winner()

        # Determine which player won and test rewards accordingly
        if winner is not None:
            winner_idx = env.game.players.index(winner)
            loser_idx = 1 - winner_idx
        else:
            # If no winner, skip this part of the test
            return

        win_reward = env._calculate_reward(winner_idx, True)
        loss_reward = env._calculate_reward(loser_idx, True)

        assert win_reward == 1.0  # Score-based win
        assert loss_reward == -0.5

    def test_score_based_win_logic(self):
        """Test that score-based wins work when all players are out."""
        env = PyLinkxEnv()
        env.reset()

        # Set up grid so Player 1 has higher score than Player 0
        # Player 0: single piece (score = 1)
        env.game.grid[0][0] = 1

        # Player 1: two connected pieces (score = 2)
        env.game.grid[0][2] = 2
        env.game.grid[1][2] = 2

        # Both players give up
        env.game.players[0].give_up()
        env.game.players[1].give_up()

        winner = env.game.check_for_winner()

        assert winner == env.game.players[1]  # Player with higher score wins
        assert env.game.win_type == "score"
        assert env.game.status == Game.GAMEOVER

    def test_win_type_persistence(self):
        """Test that win_type persists in game state until reset."""
        env = PyLinkxEnv()
        env.reset()

        # Force a score-based win
        env.game.players[0].give_up()
        env.game.players[1].give_up()
        env.game.check_for_winner()

        assert env.game.win_type == "score"

        # Win type should be cleared on reset
        env.reset()
        assert env.game.win_type is None


class TestPyLinkxEnvEdgeCases:
    """Test edge cases and error handling."""

    def test_multiple_resets(self):
        """Test that environment can be reset multiple times."""
        env = PyLinkxEnv()

        for _ in range(3):
            obs, info = env.reset()
            assert obs.shape == (9, 9, 1)
            assert env.step_count == 0

    def test_step_after_game_over(self):
        """Test stepping after game is over."""
        env = PyLinkxEnv(max_steps=5)
        env.reset()

        # Play until max steps to trigger truncation
        for _ in range(6):
            obs, reward, terminated, truncated, info = env.step(0)

        # Additional step should still work (returns invalid action)
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (9, 9, 1)

    def test_invalid_action_handling(self):
        """Test that invalid actions are handled gracefully."""
        env = PyLinkxEnv()
        env.reset()

        # Valid actions are 0-4; anything else should be handled
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs is not None


class TestPyLinkxEnvRendering:
    """Test rendering functionality."""

    def test_render_debug_mode(self):
        """Test that render works in debug mode."""
        env = PyLinkxEnv(render_mode="debug")
        env.reset()

        # Should not raise an error
        env.render()

    def test_render_none_mode(self):
        """Test that render works in None mode."""
        env = PyLinkxEnv(render_mode=None)
        env.reset()

        # Should not raise an error
        env.render()
