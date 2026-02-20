# Project Guide for PyLinkx Game

## Project Overview
PyLinkx is a tetris-like game developed using Python. The project uses object-oriented programming to manage game logic, players, and pieces. It includes a simple command-line interface for gameplay.

### Key Technologies Used
- Python 3.x
- Object-oriented Programming (OOP)
- Command-line Interface (CLI)

### High-level Architecture
The architecture consists of three main components:
1. **Game**: Manages the overall game state and flow.
2. **Player**: Represents each player's attributes and behaviors.
3. **Piece**: Manages the attributes and movements of game pieces.

## Getting Started
### Prerequisites
- Python 3.x installed on your system

### Installation Instructions
1. Clone the repository to your local machine:
   ```sh
   git clone https://github.com/yourusername/pylinkx.git
   cd pylinkx
   ```
2. Install dependencies (if any):
   ```sh
   pip install -r requirements.txt
   ```

### Basic Usage Examples
To run the game, execute the following command:
```sh
python src/main.py
```

### Running Tests
To run tests, use the following commands:
```sh
pytest tests/
```

## Project Structure
### Main Directories and Their Purpose
- **src/**: Contains the source code for the game.
  - **game.py**: Main game logic.
  - **player.py**: Player attributes and behaviors.
  - **piece.py**: Piece attributes and movements.
  - **game_renderer.py**: Handles rendering of the game on a Pygame window.

### Key Files and Their Roles
- **game.py**:
  - Manages the overall game state, including initialization, reset, and game logic.
  - Uses `GameRenderer` for rendering.
- **player.py**:
  - Defines player attributes such as name, value, color, and pieces.
  - Implements methods for player actions like giving up and checking if a player has won.
- **piece.py**:
  - Manages piece attributes like shape, position, and size.
  - Provides methods for moving, rotating, and flipping pieces.
- **game_renderer.py**:
  - Handles rendering the game on a Pygame window.
  - Draws the board, grid, pieces, and scores.

### Important Configuration Files
- **requirements.txt**: Lists project dependencies (if any).

## Development Workflow
### Coding Standards or Conventions
- Follow PEP 8 style guide for Python code.
- Use meaningful variable names.
- Write docstrings for public methods and functions.

### Testing Approach
- Write unit tests for each component using pytest.
- Ensure test coverage is at least 90%.

### Build and Deployment Process
- No build or deployment steps are required for this project.

### Contribution Guidelines
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Submit a pull request with a clear description of changes.

## Key Concepts
- **Game**: Manages the game state, player turns, and scoring.
- **Player**: Represents each player's attributes and actions.
- **Piece**: Defines the shape and movement logic of game pieces.
- **Renderer**: Handles rendering of the game on a Pygame window.


## Troubleshooting
### Common Issues and Their Solutions
- **Game crashes on startup**: Verify that all dependencies are installed and up-to-date.

### Debugging Tips
- Use print statements or a debugger to trace the execution flow.
- Inspect variables to understand their values at different points in time.


