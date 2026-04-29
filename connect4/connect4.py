"""Connect 4 — 7-column × 6-row vertical 4-in-a-row.

Same duck-typed interface as TicTacToe / ChessGame so it plugs into the
existing MCTS, model, and training loop without any of those needing changes.
Action = column index 0..6; piece falls to the lowest empty row.
Row 0 is the top of the board, row 5 is the bottom.
"""

import numpy as np


class Connect4:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.num_channels = 3  # current player, opponent, empty

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count), dtype=np.int8)

    def update_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        """Drop a piece into column `action`. Mutates and returns state."""
        col = action
        for row in range(self.row_count - 1, -1, -1):
            if state[row, col] == 0:
                state[row, col] = player
                return state
        raise ValueError(f"Column {col} is full")

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        """A column is playable iff its top cell is empty."""
        return (state[0] == 0).astype(np.uint8)

    def check_win(self, state: np.ndarray, action: int) -> bool:
        """Did the piece most recently dropped into column `action` complete a 4-in-a-row?"""
        col = action
        # Topmost piece in the column = the one just dropped.
        row = None
        for r in range(self.row_count):
            if state[r, col] != 0:
                row = r
                break
        if row is None:
            return False
        player = state[row, col]
        # Four directions: horizontal, vertical, both diagonals.
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < self.row_count and 0 <= c < self.column_count and state[r, c] == player:
                count += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while 0 <= r < self.row_count and 0 <= c < self.column_count and state[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= 4:
                return True
        return False

    def get_value_and_terminated(self, state: np.ndarray, action: int):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player: int) -> int:
        return -player

    def state_hash(self, state: np.ndarray) -> int:
        return hash(state.tobytes())

    def encode_state(self, state: np.ndarray, player: int) -> np.ndarray:
        """3 channels from `player`'s perspective: own, opponent, empty."""
        encoded = np.zeros((3, self.row_count, self.column_count), dtype=np.float32)
        encoded[0] = (state == player).astype(np.float32)
        encoded[1] = (state == -player).astype(np.float32)
        encoded[2] = (state == 0).astype(np.float32)
        return encoded
