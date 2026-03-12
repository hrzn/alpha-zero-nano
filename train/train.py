"""Self-play and training loop for AlphaZero."""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from mcts.mcts import MCTS


def self_play(game, mcts, model):
    """Play one game via MCTS+model and return training examples.

    Returns a list of (encoded_state, policy, outcome) tuples where:
    - encoded_state: (3, rows, cols) float32 array from the acting player's perspective
    - policy: (action_size,) float32 array of MCTS visit-count probabilities
    - outcome: float in {-1, 0, 1} from the acting player's perspective
    """
    examples = []  # (encoded_state, policy, player)
    state = game.get_initial_state()
    player = 1

    while True:
        policy = mcts.search(state, player)
        encoded_state = model.encode_state(state, player)
        examples.append((encoded_state, policy, player))

        action = np.random.choice(game.action_size, p=policy)
        state = game.update_state(state, action, player)

        value, terminated = game.get_value_and_terminated(state, action)
        if terminated:
            # Assign outcomes: value=1 means the player who just moved won
            training_examples = []
            for enc_state, pol, acting_player in examples:
                if value == 0:
                    outcome = 0.0
                elif acting_player == player:
                    # This player just won
                    outcome = 1.0
                else:
                    # This player lost
                    outcome = -1.0
                training_examples.append((enc_state, pol, outcome))
            return training_examples

        player = game.get_opponent(player)


def train_step(model, optimizer, batch):
    """Run one gradient update step on a batch of training examples.

    Args:
        model: ResNet instance
        optimizer: torch optimizer
        batch: (encoded_states, policies, outcomes) as numpy arrays

    Returns:
        loss value as a float
    """
    model.train()
    encoded_states, policies, outcomes = batch

    states_t = torch.tensor(encoded_states, dtype=torch.float32)
    policies_t = torch.tensor(policies, dtype=torch.float32)
    outcomes_t = torch.tensor(outcomes, dtype=torch.float32)

    policy_logits, values = model(states_t)

    policy_loss = F.cross_entropy(policy_logits, policies_t)
    value_loss = F.mse_loss(values.squeeze(-1), outcomes_t)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


class AlphaZero:
    def __init__(self, game, model, args):
        """
        args keys:
          num_searches      — MCTS simulations per move
          num_self_play_games — games per iteration
          num_epochs        — training epochs per iteration
          batch_size        — examples per gradient step
          lr                — learning rate
        """
        self.game = game
        self.model = model
        self.args = args
        self.mcts = MCTS(game, model=model, num_searches=args["num_searches"])
        self.optimizer = optim.Adam(model.parameters(), lr=args["lr"])

    def run(self, num_iterations):
        """Run the full AlphaZero training loop."""
        for _ in range(num_iterations):
            # Self-play: collect examples from all games this iteration
            examples = []
            for _ in range(self.args["num_self_play_games"]):
                examples += self_play(self.game, self.mcts, self.model)

            # Unpack into arrays
            encoded_states = np.array([e[0] for e in examples], dtype=np.float32)
            policies = np.array([e[1] for e in examples], dtype=np.float32)
            outcomes = np.array([e[2] for e in examples], dtype=np.float32)

            # Train for num_epochs, sampling a random batch each epoch
            self.model.train()
            for _ in range(self.args["num_epochs"]):
                batch_size = min(self.args["batch_size"], len(examples))
                idx = np.random.choice(len(examples), size=batch_size, replace=False)
                batch = (encoded_states[idx], policies[idx], outcomes[idx])
                train_step(self.model, self.optimizer, batch)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
