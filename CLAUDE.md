# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

This is a from-scratch implementation of the AlphaZero algorithm applied to simple board games, starting with Tic-tac-toe, all the way to chess. The goal is to build a minimal, educational implementation, which ultimately we hope to be able to train locally on a Macbook pro. We are not interested in targeting state-of-the-art performance; but would be extremely interested in understanding how far/good this can be made while training only on a local machine for now.

## Commands

This project uses `uv` for Python package management (Python 3.12).

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run a single test
uv run pytest tictactoe/test_tictactoe.py::test_check_win

# Run the Jupyter notebook (interactive game loop)
uv run jupyter notebook az-tictactoe.ipynb
```

## Architecture

The project is structured around game environment classes that implement a consistent interface, used by the AlphaZero algorithm components.

**`tictactoe/tictactoe.py` — `TicTacToe` class**
The game environment. Board state is a `numpy` array where player 1 = `1`, player 2 = `-1`, empty = `0`. Actions are integers `0–8` mapping to board positions row-major. Key methods:
- `get_initial_state()` → zeroed 3×3 numpy array
- `update_state(state, action, player)` → mutates and returns state
- `get_valid_moves(state)` → uint8 mask of length `action_size`
- `check_win(state, action)` → bool, checks the last action for a win
- `get_value_and_terminated(state, action)` → `(value, done)` tuple; value is `1` for win, `0` for draw/ongoing
- `get_opponent(player)` → negates player (alternates between `1` and `-1`)

**`az-tictactoe.ipynb`** — Interactive notebook for playing Tic-tac-toe manually via the game loop, and will grow to include MCTS and neural network training.

See `STRATEGY.md` for the full implementation plan covering MCTS, neural network (PyTorch ResNet), and self-play training — designed to be game-agnostic and scale from Tic-tac-toe to chess.
