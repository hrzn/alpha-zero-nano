# alpha-zero-nano

A (somewhat) minimal vibe-coded implementation of alpha-zero style agents for 2-player games. Successfully trainable on a Macbook pro for tictactoe (trivial) and Connect-4 (easy). Chess would require more compute. The training is using self-play only, following the alpha-zero recipe. 

The repo contains a generic training loop and MCTS, with different pluggable network and game modules, for different games. Currently tictactoe, Connect-4 and chess are implemented.

For Connect-4, there's also a small webapp allowing to play against the agent, visualse the policy distribution and predicted value from the network, as well as the distribution from MCTS.

Explore it at: https://hrzn.github.io/alpha-zero-nano/



## Training
```
uv run python -m train.run_training --preset C4
```

## Exporting a trained Connect-4 model for web:
```
uv run python tools/export_for_web.py
```

## More details
The implementation does contain a few not-so-minimal ingredients:
* MCTS tree memoization
* parallel self play
* batched MCTS inference
* "arena gating", by which models are kept across iterations only if they provide an improvement, to avoid regressions