import pytest
from tictactoe import TicTacToe
import numpy as np

@pytest.fixture
def tictactoe():
    return TicTacToe()

def test_action_to_row_col(tictactoe):
    assert tictactoe.action_to_row_col(0) == (0, 0)
    assert tictactoe.action_to_row_col(1) == (0, 1)
    assert tictactoe.action_to_row_col(2) == (0, 2)
    assert tictactoe.action_to_row_col(3) == (1, 0)
    assert tictactoe.action_to_row_col(4) == (1, 1)
    assert tictactoe.action_to_row_col(5) == (1, 2)
    assert tictactoe.action_to_row_col(6) == (2, 0)
    assert tictactoe.action_to_row_col(7) == (2, 1)
    assert tictactoe.action_to_row_col(8) == (2, 2)

def test_check_win(tictactoe):
    state = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    assert tictactoe.check_win(state, 0) == True
    state = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]])
    assert tictactoe.check_win(state, 6) == True
    state = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert tictactoe.check_win(state, 4) == True
    state = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert tictactoe.check_win(state, 8) == True
    state = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert tictactoe.check_win(state, 2) == False

def test_get_value_and_terminated(tictactoe):
    state = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    assert tictactoe.get_value_and_terminated(state, 0) == (1, True)
