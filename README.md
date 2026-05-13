# alpha-zero-nano

A minimal vibe-coded implementation of alpha-zero style agents for 2-player games. Successfully trainable on a Macbook pro for tictactoe (trivial) and Connect-4 (easy). Chess would require more compute. The training is using self-play only, following the alpha-zero recipe. 

The repo contains a generic training loop and MCTS, with different pluggable network and game modules, for different games. Currently tictactoe, Connect-4 and chess are implemented.

For Connect-4, there's also a small webapp allowing to play against the agent, visualse the policy distribution and predicted value from the network, as well as the distribution from MCTS.