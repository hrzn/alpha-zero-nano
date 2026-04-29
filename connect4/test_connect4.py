import numpy as np
import pytest

from connect4 import Connect4


@pytest.fixture
def c4():
    return Connect4()


def test_attrs(c4):
    assert c4.row_count == 6
    assert c4.column_count == 7
    assert c4.action_size == 7
    assert c4.num_channels == 3


def test_initial_state(c4):
    state = c4.get_initial_state()
    assert state.shape == (6, 7)
    assert np.all(state == 0)


def test_gravity(c4):
    state = c4.get_initial_state()
    c4.update_state(state, 3, 1)
    assert state[5, 3] == 1  # bottom row
    c4.update_state(state, 3, -1)
    assert state[4, 3] == -1  # stacks on top
    c4.update_state(state, 3, 1)
    assert state[3, 3] == 1


def test_valid_moves_full_column(c4):
    state = c4.get_initial_state()
    for _ in range(c4.row_count):
        c4.update_state(state, 0, 1)
    valid = c4.get_valid_moves(state)
    assert valid[0] == 0  # column 0 full
    assert valid.sum() == 6  # other 6 columns playable


def test_full_column_raises(c4):
    state = c4.get_initial_state()
    for _ in range(c4.row_count):
        c4.update_state(state, 0, 1)
    with pytest.raises(ValueError):
        c4.update_state(state, 0, 1)


def test_horizontal_win(c4):
    state = c4.get_initial_state()
    # Player 1 fills bottom row columns 0-3
    for col in [0, 1, 2, 3]:
        c4.update_state(state, col, 1)
    assert c4.check_win(state, 3) is True
    value, terminated = c4.get_value_and_terminated(state, 3)
    assert (value, terminated) == (1, True)


def test_vertical_win(c4):
    state = c4.get_initial_state()
    for _ in range(4):
        c4.update_state(state, 2, 1)
    assert c4.check_win(state, 2) is True


def test_diagonal_down_right_win(c4):
    # Build a \ diagonal of 1's at (2,0),(3,1),(4,2),(5,3)
    state = c4.get_initial_state()
    state[5, 0] = 1
    state[5, 1] = -1; state[4, 1] = 1
    state[5, 2] = -1; state[4, 2] = -1; state[3, 2] = 1
    state[5, 3] = -1; state[4, 3] = -1; state[3, 3] = -1; state[2, 3] = 1
    assert c4.check_win(state, 3) is True


def test_diagonal_down_left_win(c4):
    # / diagonal at (5,0),(4,1),(3,2),(2,3)
    state = c4.get_initial_state()
    state[5, 0] = -1; state[4, 0] = -1; state[3, 0] = -1; state[2, 0] = 1
    state[5, 1] = -1; state[4, 1] = -1; state[3, 1] = 1
    state[5, 2] = -1; state[4, 2] = 1
    state[5, 3] = 1
    assert c4.check_win(state, 3) is True


def test_no_win(c4):
    state = c4.get_initial_state()
    c4.update_state(state, 3, 1)
    assert c4.check_win(state, 3) is False
    assert c4.get_value_and_terminated(state, 3) == (0, False)


def test_draw_when_full_no_winner(c4):
    """Build a known full-board no-winner position and verify draw."""
    # Pattern that fills the board without 4-in-a-row.
    # Bottom-up, we use a shifted column pattern.
    state = np.array([
        [ 1,-1, 1,-1, 1,-1, 1],
        [ 1,-1, 1,-1, 1,-1, 1],
        [-1, 1,-1, 1,-1, 1,-1],
        [-1, 1,-1, 1,-1, 1,-1],
        [ 1,-1, 1,-1, 1,-1, 1],
        [ 1,-1, 1,-1, 1,-1, 1],
    ], dtype=np.int8)
    # Confirm it has no 4-in-a-row (sanity).
    c4_inst = Connect4()
    for col in range(7):
        assert c4_inst.check_win(state, col) is False
    # Board is full → draw.
    assert c4_inst.get_valid_moves(state).sum() == 0
    # value_and_terminated treats it as a draw regardless of `action`.
    value, terminated = c4_inst.get_value_and_terminated(state, 0)
    assert (value, terminated) == (0, True)


def test_get_opponent(c4):
    assert c4.get_opponent(1) == -1
    assert c4.get_opponent(-1) == 1


def test_state_hash_consistency(c4):
    s1 = c4.get_initial_state()
    s2 = c4.get_initial_state()
    assert c4.state_hash(s1) == c4.state_hash(s2)
    c4.update_state(s1, 3, 1)
    assert c4.state_hash(s1) != c4.state_hash(s2)


def test_encode_state_perspective(c4):
    state = c4.get_initial_state()
    c4.update_state(state, 3, 1)   # player 1 piece at (5,3)
    c4.update_state(state, 4, -1)  # player -1 piece at (5,4)

    enc1 = c4.encode_state(state, 1)
    assert enc1.shape == (3, 6, 7)
    assert enc1[0, 5, 3] == 1.0  # own
    assert enc1[1, 5, 4] == 1.0  # opp
    assert enc1[2, 0, 0] == 1.0  # empty

    # From the opposite perspective, own/opp swap.
    enc2 = c4.encode_state(state, -1)
    assert enc2[0, 5, 4] == 1.0
    assert enc2[1, 5, 3] == 1.0


def test_encode_state_dtype(c4):
    state = c4.get_initial_state()
    enc = c4.encode_state(state, 1)
    assert enc.dtype == np.float32
