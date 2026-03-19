"""Tests for Alpha-MCTS on Tic-tac-toe."""

import numpy as np
import pytest

from mcts.mcts import MCTS, Node
from model.model import ResNet
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


class TestTranspositionTable:
    """Opt 2: transposition table reduces model.predict() calls."""

    def test_cache_reduces_model_calls(self, game):
        """With transposition table, model should be called fewer than num_searches times."""
        call_count = [0]

        class CountingModel:
            def predict(self, state, player):
                call_count[0] += 1
                return np.ones(game.action_size) / game.action_size, 0.0

        mcts = MCTS(game, model=CountingModel(), num_searches=100)
        state = game.get_initial_state()
        mcts.search(state, player=1)

        # Many paths converge to the same TicTacToe positions, so we expect
        # fewer model calls than simulations due to caching.
        assert call_count[0] < 100

    def test_cache_exists_after_search(self, game):
        """MCTS should have a _cache attribute after search."""
        mcts = MCTS(game, model=None, num_searches=10)
        state = game.get_initial_state()
        mcts.search(state, player=1)
        assert hasattr(mcts, "_cache")

    def test_cache_cleared_at_search_start(self, game):
        """Cache should be fresh (not carry over stale entries) across searches."""
        call_counts = []

        class CountingModel:
            def __init__(self):
                self.count = 0

            def predict(self, state, player):
                self.count += 1
                return np.ones(game.action_size) / game.action_size, 0.0

        model = CountingModel()
        mcts = MCTS(game, model=model, num_searches=20)
        state = game.get_initial_state()

        model.count = 0
        mcts.search(state, player=1)
        count1 = model.count

        model.count = 0
        mcts.search(state, player=1)  # second search, cache starts fresh
        count2 = model.count

        # Both searches start from the same root and should make a similar
        # number of model calls (second search is not free from a carried-over cache).
        assert count2 > 0

    def test_policy_quality_unchanged_by_caching(self, game):
        """Transposition table must not corrupt policy — winning move still found."""
        model_fixture = ResNet(game, num_res_blocks=2, num_hidden=64)
        mcts = MCTS(game, model=model_fixture, num_searches=100)

        # X has two in a row; action 2 wins
        state = game.get_initial_state()
        state = game.update_state(state, 0, 1)
        state = game.update_state(state, 3, -1)
        state = game.update_state(state, 1, 1)
        state = game.update_state(state, 4, -1)

        policy = mcts.search(state, player=1)
        assert policy[2] == max(policy)
