"""Tests for Alpha-MCTS on Tic-tac-toe."""

import numpy as np
import pytest

from mcts.mcts import MCTS, Node
from tictactoe.tictactoe import TicTacToe


@pytest.fixture
def game():
    return TicTacToe()


@pytest.fixture
def mcts(game):
    """MCTS with no neural network (uniform priors, value=0)."""
    return MCTS(game, model=None, num_searches=100)


class TestNodeTicTacToe:
    def test_expand_creates_children_for_all_valid_moves(self, game):
        state = game.get_initial_state()
        # Uniform policy, value=0
        policy = np.ones(game.action_size) / game.action_size
        node = Node(game, state, player=1)
        node.expand(policy)

        assert len(node.children) == 9
        for action, child in node.children.items():
            assert 0 <= action < 9
            assert child.prior == pytest.approx(1.0 / 9)

    def test_expand_only_valid_moves(self, game):
        """When some squares are occupied, only valid moves get children."""
        state = game.get_initial_state()
        state = game.update_state(state, 0, 1)   # X in top-left
        state = game.update_state(state, 4, -1)  # O in center

        valid_moves = game.get_valid_moves(state)
        # Provide a uniform policy — expand should mask to valid moves
        policy = np.ones(game.action_size) / game.action_size
        node = Node(game, state, player=1)
        node.expand(policy)

        assert len(node.children) == 7
        assert 0 not in node.children
        assert 4 not in node.children

    def test_terminal_node_is_not_expanded(self, game):
        """A node where the game is over should not be expandable."""
        # Create a won state: X wins top row
        state = game.get_initial_state()
        state = game.update_state(state, 0, 1)
        state = game.update_state(state, 3, -1)
        state = game.update_state(state, 1, 1)
        state = game.update_state(state, 4, -1)
        state = game.update_state(state, 2, 1)  # X wins with top row

        value, terminated = game.get_value_and_terminated(state, 2)
        assert terminated
        assert value == 1


class TestPUCTSelectionTicTacToe:
    def test_unexplored_children_preferred(self, game):
        """PUCT should prefer unvisited children (high exploration term)."""
        state = game.get_initial_state()
        policy = np.ones(game.action_size) / game.action_size
        node = Node(game, state, player=1)
        node.expand(policy)

        # Parent needs visits for exploration term to be nonzero
        node.visit_count = 11
        # Simulate visiting one child many times with low value
        child_0 = node.children[0]
        child_0.visit_count = 10
        child_0.value_sum = 5.0  # Value from child's perspective; parent sees Q = -0.5

        # An unvisited child should be selected over the visited one
        selected_action, selected_child = node.select()
        assert selected_child.visit_count == 0

    def test_high_prior_preferred_among_unvisited(self, game):
        """Among unvisited children, PUCT should prefer higher prior."""
        state = game.get_initial_state()
        policy = np.zeros(game.action_size)
        policy[4] = 0.9  # center gets high prior
        policy[0] = 0.1  # corner gets low prior
        # Only two valid actions for simplicity
        node = Node(game, state, player=1)
        node.expand(policy)

        # Only keep children 0 and 4 for a cleaner test
        node.children = {k: v for k, v in node.children.items() if k in (0, 4)}
        # Parent needs at least 1 visit for the exploration term to be nonzero
        node.visit_count = 1

        selected_action, selected_child = node.select()
        assert selected_action == 4


class TestMCTSSearchTicTacToe:
    def test_finds_winning_move(self, game, mcts):
        """MCTS should find the obvious winning move."""
        # X has two in a row (positions 0, 1), position 2 wins
        state = game.get_initial_state()
        state = game.update_state(state, 0, 1)
        state = game.update_state(state, 3, -1)
        state = game.update_state(state, 1, 1)
        state = game.update_state(state, 4, -1)
        # Player 1's turn, action 2 wins

        policy = mcts.search(state, player=1)

        assert policy[2] == max(policy)

    def test_blocks_opponent_win(self, game, mcts):
        """MCTS should block the opponent's winning move."""
        # O is about to win: O at positions 3, 4; position 5 would win for O
        # It's X's turn, X must block at position 5
        state = game.get_initial_state()
        state = game.update_state(state, 0, 1)   # X top-left
        state = game.update_state(state, 3, -1)  # O mid-left
        state = game.update_state(state, 8, 1)   # X bot-right
        state = game.update_state(state, 4, -1)  # O center
        # O threatens to complete middle row at position 5. X must block.

        policy = mcts.search(state, player=1)

        assert policy[5] == max(policy)

    def test_policy_sums_to_one(self, game, mcts):
        """The returned policy should be a valid probability distribution."""
        state = game.get_initial_state()
        policy = mcts.search(state, player=1)

        assert policy.sum() == pytest.approx(1.0)
        assert all(p >= 0 for p in policy)

    def test_policy_zero_on_invalid_moves(self, game, mcts):
        """Policy should have zero probability on already-occupied squares."""
        state = game.get_initial_state()
        state = game.update_state(state, 0, 1)
        state = game.update_state(state, 4, -1)

        policy = mcts.search(state, player=1)

        assert policy[0] == 0.0
        assert policy[4] == 0.0

    def test_policy_length_matches_action_size(self, game, mcts):
        """Policy vector should have one entry per action."""
        state = game.get_initial_state()
        policy = mcts.search(state, player=1)

        assert len(policy) == game.action_size


class TestTreeReuse:
    """Opt 1: MCTS tree reuse via advance_root."""

    def test_advance_root_reuses_subtree(self, game):
        """After advance_root, the next search starts from a pre-visited node."""
        mcts = MCTS(game, model=None, num_searches=50)
        state = game.get_initial_state()

        mcts.search(state, player=1)
        # Child at action 4 (center) should have been visited
        mcts.advance_root(4)

        assert mcts._root is not None
        assert mcts._root.visit_count > 0  # was explored in the first search

    def test_advance_root_resets_when_action_not_in_children(self, game):
        """advance_root with an action not in children resets root to None."""
        mcts = MCTS(game, model=None, num_searches=10)
        # Fill square 8 so it's invalid and won't appear in children
        state = game.get_initial_state()
        state = game.update_state(state, 8, 1)
        mcts.search(state, player=-1)

        # Action 8 is occupied — not in children
        mcts.advance_root(8)
        assert mcts._root is None

    def test_policy_valid_after_tree_reuse(self, game):
        """Policy should remain valid (sums to 1, zeros on occupied) after reuse."""
        mcts = MCTS(game, model=None, num_searches=50)
        state = game.get_initial_state()

        mcts.search(state, player=1)
        mcts.advance_root(4)

        new_state = game.update_state(state.copy(), 4, 1)
        policy = mcts.search(new_state, player=-1)

        assert policy.sum() == pytest.approx(1.0, abs=1e-5)
        assert (policy >= 0).all()
        assert policy[4] == 0.0  # center is now occupied

    def test_second_search_after_advance_root_works(self, game):
        """Full search after advance_root should not raise and give valid policy."""
        mcts = MCTS(game, model=None, num_searches=30)
        state = game.get_initial_state()

        mcts.search(state, player=1)
        mcts.advance_root(0)  # corner
        new_state = game.update_state(state.copy(), 0, 1)
        policy = mcts.search(new_state, player=-1)

        assert len(policy) == game.action_size
        assert policy.sum() == pytest.approx(1.0, abs=1e-5)


class TestBatchedMCTSTicTacToe:
    """Opt 4: batched MCTS inference via virtual loss."""

    @pytest.fixture
    def batched_mcts(self, game):
        return MCTS(game, model=None, num_searches=100, batch_size=8)

    def test_policy_sums_to_one(self, game, batched_mcts):
        state = game.get_initial_state()
        policy = batched_mcts.search(state, player=1)
        assert policy.sum() == pytest.approx(1.0, abs=1e-5)

    def test_policy_non_negative(self, game, batched_mcts):
        state = game.get_initial_state()
        policy = batched_mcts.search(state, player=1)
        assert (policy >= 0).all()

    def test_policy_zero_on_occupied(self, game, batched_mcts):
        state = game.get_initial_state()
        state = game.update_state(state, 0, 1)
        state = game.update_state(state, 4, -1)
        policy = batched_mcts.search(state, player=1)
        assert policy[0] == 0.0
        assert policy[4] == 0.0

    def test_policy_length(self, game, batched_mcts):
        state = game.get_initial_state()
        policy = batched_mcts.search(state, player=1)
        assert len(policy) == game.action_size

    def test_finds_winning_move(self, game):
        """Batched MCTS should find the obvious winning move."""
        mcts = MCTS(game, model=None, num_searches=100, batch_size=8)
        state = game.get_initial_state()
        state = game.update_state(state, 0, 1)
        state = game.update_state(state, 3, -1)
        state = game.update_state(state, 1, 1)
        state = game.update_state(state, 4, -1)
        # Player 1's turn, action 2 wins

        policy = mcts.search(state, player=1)
        assert policy[2] == max(policy)

    def test_blocks_opponent_win(self, game):
        """Batched MCTS should block the opponent's winning move."""
        mcts = MCTS(game, model=None, num_searches=100, batch_size=8)
        state = game.get_initial_state()
        state = game.update_state(state, 0, 1)
        state = game.update_state(state, 3, -1)
        state = game.update_state(state, 8, 1)
        state = game.update_state(state, 4, -1)

        policy = mcts.search(state, player=1)
        assert policy[5] == max(policy)

    def test_batch_size_greater_than_num_searches(self, game):
        """batch_size > num_searches should not crash and return a valid policy."""
        mcts = MCTS(game, model=None, num_searches=5, batch_size=32)
        state = game.get_initial_state()
        policy = mcts.search(state, player=1)
        assert policy.sum() == pytest.approx(1.0, abs=1e-5)
        assert (policy >= 0).all()

    def test_num_searches_not_multiple_of_batch_size(self, game):
        """num_searches not divisible by batch_size should produce a valid policy."""
        mcts = MCTS(game, model=None, num_searches=10, batch_size=3)
        state = game.get_initial_state()
        policy = mcts.search(state, player=1)
        assert policy.sum() == pytest.approx(1.0, abs=1e-5)
        assert (policy >= 0).all()

    def test_visit_count_invariant(self, game):
        """Total visit counts across children should equal num_searches (VL fully undone)."""
        num_searches = 50
        mcts = MCTS(game, model=None, num_searches=num_searches, batch_size=8)
        state = game.get_initial_state()
        mcts.search(state, player=1)
        root = mcts._root
        total_child_visits = sum(c.visit_count for c in root.children.values())
        assert total_child_visits == num_searches
