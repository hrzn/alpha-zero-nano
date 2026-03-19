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
        self.num_res_blocks = num_res_blocks
        self.num_hidden = num_hidden

        # Input channels come from the game's encoding
        self.input_block = nn.Sequential(
            nn.Conv2d(game.num_channels, num_hidden, kernel_size=3, padding=1),
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

    @torch.no_grad()
    def predict(self, state, player: int):
        """Run inference on a single state. Returns (policy, value) as numpy."""
        self.eval()
        device = next(self.parameters()).device
        encoded = self.game.encode_state(state, player)
        x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(device)
        policy_logits, value = self(x)
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        return policy, float(value.item())
