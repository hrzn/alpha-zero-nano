"""Implements a basic Tic tac toe game."""

import numpy as np


class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count
        self.num_channels = 3  # current player, opponent, empty

    def action_to_row_col(self, action: int) -> tuple[int, int]:
        """Transforms an integer action into a (row, col) tuple."""
        row = action // self.column_count
        column = action % self.column_count
        return row, column

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def update_state(self, state: np.ndarray, action: int, player) -> np.ndarray:
        """Mutates the state to a new state, given an action and a player."""
        row, column = self.action_to_row_col(action)
        state[row, column] = player
        return state
    
    def get_valid_moves(self, state: np.ndarray):
        """Returns valid actions from a given state."""
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state: np.ndarray, action: int):
        """Check whether this last action (yielding `state`) was winning."""
        row, column = self.action_to_row_col(action)
        player = state[row, column]
        return (
            np.all(state[row, :] == player) 
            or np.all(state[:, column] == player) 
            or np.all(np.diag(state) == player)
            or np.all(np.diag(np.fliplr(state)) == player)
        )
    
    def get_value_and_terminated(self, state: np.ndarray, action: int):
        """Return whether there was as win, and whether game is terminated."""
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player

    def state_hash(self, state: np.ndarray) -> int:
        return hash(state.tobytes())

    def encode_state(self, state: np.ndarray, player: int) -> np.ndarray:
        """Encode board as 3 channels from the given player's perspective.

        Channel 0: current player's pieces
        Channel 1: opponent's pieces
        Channel 2: empty squares
        """
        encoded = np.zeros((3, self.row_count, self.column_count), dtype=np.float32)
        encoded[0] = (state == player).astype(np.float32)
        encoded[1] = (state == -player).astype(np.float32)
        encoded[2] = (state == 0).astype(np.float32)
        return encoded
