"""ResNet model with policy and value heads for AlphaZero."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class ResNet(nn.Module):
    def __init__(self, game, num_res_blocks, num_hidden):
        super().__init__()
        self.game = game

        # Input: 3 channels (current player pieces, opponent pieces, empty)
        self.input_block = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.res_blocks = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """x: (batch, 3, rows, cols) — returns (policy_logits, value)."""
        x = self.input_block(x)
        for block in self.res_blocks:
            x = block(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy_logits, value

    def encode_state(self, state: np.ndarray, player: int) -> np.ndarray:
        """Encode a board state into 3 channels from the given player's perspective.

        Channel 0: current player's pieces
        Channel 1: opponent's pieces
        Channel 2: empty squares
        """
        encoded = np.zeros((3, self.game.row_count, self.game.column_count), dtype=np.float32)
        encoded[0] = (state == player).astype(np.float32)
        encoded[1] = (state == -player).astype(np.float32)
        encoded[2] = (state == 0).astype(np.float32)
        return encoded

    @torch.no_grad()
    def predict(self, state: np.ndarray, player: int):
        """Run inference on a single state. Returns (policy, value) as numpy."""
        self.eval()
        encoded = self.encode_state(state, player)
        x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        policy_logits, value = self(x)
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).numpy()
        return policy, float(value.item())
