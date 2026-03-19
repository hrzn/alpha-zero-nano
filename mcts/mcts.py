"""AlphaZero-style Monte Carlo Tree Search with PUCT selection.

Optimizations implemented:
  Opt 1 — Tree reuse: advance_root() promotes a child to root so the next
           search() call reuses its visit counts instead of starting fresh.
"""

import math

import numpy as np


class Node:
    def __init__(self, game, state, player, prior=0.0, parent=None, action_taken=None):
        self.game = game
        self.state = state
        self.player = player
        self.prior = prior
        self.parent = parent
        self.action_taken = action_taken

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def is_expanded(self):
        return len(self.children) > 0

    def select(self, c_puct=1.0):
        """Select the child with the highest PUCT score."""
        best_score = -float("inf")
        best_action = None
        best_child = None

        for action, child in self.children.items():
            score = child.puct_score(c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def puct_score(self, c_puct):
        # Negate because value_sum is from this node's player perspective,
        # but the parent (who is selecting) is the opponent
        q_value = 0.0 if self.visit_count == 0 else -self.value_sum / self.visit_count
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return q_value + exploration

    def expand(self, policy):
        """Create child nodes for each valid move, using policy as priors."""
        valid_moves = self.game.get_valid_moves(self.state)
        # Mask and renormalize policy to valid moves
        policy = policy * valid_moves
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum

        for action in range(self.game.action_size):
            if valid_moves[action] == 1:
                child_state = self.state.copy()
                child_state = self.game.update_state(child_state, action, self.player)
                child = Node(
                    game=self.game,
                    state=child_state,
                    player=self.game.get_opponent(self.player),
                    prior=policy[action],
                    parent=self,
                    action_taken=action,
                )
                self.children[action] = child

    def backpropagate(self, value):
        """Propagate value up to root, flipping sign at each level."""
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    def __init__(self, game, model=None, num_searches=100, c_puct=1.0):
        self.game = game
        self.model = model
        self.num_searches = num_searches
        self.c_puct = c_puct
        self._root = None  # Opt 1: tree reuse

    def _evaluate(self, state, player):
        """Get policy and value from the model, or use defaults if no model."""
        if self.model is None:
            return np.ones(self.game.action_size) / self.game.action_size, 0.0
        return self.model.predict(state, player)

    def advance_root(self, action):
        """Opt 1: promote the child at action to be the new root.

        Call this after each move in self-play so the next search() reuses
        the subtree already explored under that child. If the action was
        not in the tree, _root is reset to None (fresh tree on next search).
        """
        if self._root is not None and action in self._root.children:
            self._root = self._root.children[action]
            self._root.parent = None
        else:
            self._root = None

    def search(self, state, player):
        """Run MCTS from the given state and return a policy vector."""
        # Opt 1: reuse existing root if available (set via advance_root),
        # otherwise build a fresh root and expand it.
        if self._root is None:
            root = Node(self.game, state.copy(), player)
            policy, _ = self._evaluate(state, player)
            root.expand(policy)
            self._root = root
        else:
            root = self._root

        for _ in range(self.num_searches):
            node = root

            # Select: traverse tree until we find an unexpanded node
            while node.is_expanded():
                _, node = node.select(self.c_puct)

            # Check if this node is terminal
            value, terminated = self.game.get_value_and_terminated(
                node.state, node.action_taken
            )

            if terminated:
                # Value is from the perspective of the player who just moved
                # (the parent's player), but we need it from this node's
                # perspective for backpropagation
                value = -value
            else:
                # Expand and evaluate
                policy, value = self._evaluate(node.state, node.player)
                node.expand(policy)

            # Backpropagate
            node.backpropagate(value)

        # Build policy from visit counts
        action_probs = np.zeros(self.game.action_size)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count

        action_probs = action_probs / action_probs.sum()
        return action_probs
