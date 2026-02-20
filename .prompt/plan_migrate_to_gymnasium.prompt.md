Plan: Migrate PyLinkx to Gymnasium RL

This plan details a step-by-step migration of PyLinkx to use Gymnasium for reinforcement learning. The goal is to wrap the game logic in a Gymnasium-compatible environment, refactor for programmatic control, and enable RL agent training. Key decisions include using Gymnasium for compatibility, supporting both single-agent and self-play modes, and starting with a sparse reward structure.

**Steps**
1. **Update Dependencies**
   - Add `gymnasium` and RL libraries (e.g., `stable-baselines3`, `pettingzoo`) to requirements.txt.
   - Remove obsolete RL packages if present.

2. **Create Gymnasium Environment Wrapper**
   - Add src/game_env.py with a `PyLinkxEnv` class inheriting from `gymnasium.Env` (or `MultiAgentEnv` for self-play).
   - Implement `reset()`, `step(action)`, `render()`, and define `observation_space` and `action_space`.

3. **Refactor Game Logic**
   - Decouple UI and player input from src/game.py and src/player.py.
   - Ensure `Game` and `Player` classes support programmatic actions and state transitions.
   - Add methods like `get_observation()` and standardize reward calculation.

4. **Multi-Agent/Self-Play Support**
   - Refactor to support agent indexing and turn management.
   - Consider PettingZoo or Gymnasium’s multi-agent API if needed.

5. **Testing and Validation**
   - Add tests for the new environment in tests/, e.g., tests/test_rl_env.py.
   - Validate `reset()`, `step()`, and reward logic.
   - Ensure existing game logic tests still pass.

6. **Training Script Integration**
   - Create src/train.py to run RL training using the new environment.
   - Integrate RL algorithms and log agent performance.

**Verification**
- Run new and existing tests for correctness.
- Use the training script to verify agent interaction and learning.
- Manually inspect episodes for correct state transitions and rewards.

**Decisions**
- Use Gymnasium for RL environment compatibility.
- Start with sparse win/loss rewards, with potential for incremental rewards.
- Support both single-agent and self-play modes.

Let me know if you want to refine any step or need a sample environment scaffold before implementation.
