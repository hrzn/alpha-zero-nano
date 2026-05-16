//! Self-play smoke: one game with a uniform-prior dummy evaluator. No NN, no
//! fixture lookup — exercises the entire algorithm (MCTS + Dirichlet at the
//! root + temperature sampling + value bootstrap) and confirms the output
//! has the expected structural shape.
//!
//! What this guards against:
//!   - MCTS or selfplay panicking on an unexpected position
//!   - Off-by-one in the temperature threshold (we set it to 0 so every
//!     move is argmax, and verify the game still terminates)
//!   - Encoded states / policies coming out the wrong shape
//!   - Value sign bookkeeping (every example must have value in [-1, 1])

use alpha_zero_nano::inference::UniformEvaluator;
use alpha_zero_nano::selfplay::{play_game, SelfPlayConfig};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn one_game_produces_well_shaped_examples() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xa1b2c3d4);
    let cfg = SelfPlayConfig {
        num_searches: 16,
        c_puct: 1.0,
        batch_size: 1,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        max_moves: 40,
        temp_threshold: Some(8),
    };
    let examples = play_game(&UniformEvaluator, &cfg, &mut rng);

    assert!(!examples.is_empty(), "self-play produced no examples");
    assert!(
        examples.len() <= cfg.max_moves as usize,
        "produced {} examples but max_moves={}",
        examples.len(),
        cfg.max_moves,
    );

    for (i, ex) in examples.iter().enumerate() {
        assert_eq!(ex.state.0.len(), 17 * 8 * 8, "ex {i} state shape");
        assert_eq!(ex.policy.len(), 4096, "ex {i} policy length");

        let policy_sum: f32 = ex.policy.iter().sum();
        assert!(
            (policy_sum - 1.0).abs() < 1e-5,
            "ex {i} policy sum {policy_sum}, want 1.0",
        );
        assert!(
            ex.policy.iter().all(|p| p.is_finite() && *p >= 0.0),
            "ex {i} has negative or NaN policy entry",
        );
        assert!(ex.value.is_finite(), "ex {i} value not finite: {}", ex.value);
        assert!(
            ex.value.abs() <= 1.0 + 1e-5,
            "ex {i} value {} outside [-1, 1]",
            ex.value,
        );
        assert!(ex.state.0.iter().all(|v| v.is_finite()), "ex {i} state has NaN/Inf");
    }
}

#[test]
fn game_under_temperature_zero_is_terminal_or_capped() {
    // temp_threshold=Some(0) means we go argmax from move 0. With uniform
    // priors and 16 MCTS sims, the same first action is picked repeatedly
    // unless the game ends — useful to verify the loop reaches a terminal.
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let cfg = SelfPlayConfig {
        num_searches: 16,
        c_puct: 1.0,
        batch_size: 1,
        dirichlet_alpha: 0.0,
        dirichlet_epsilon: 0.0,
        max_moves: 60,
        temp_threshold: Some(0),
    };
    let examples = play_game(&UniformEvaluator, &cfg, &mut rng);
    assert!(!examples.is_empty());
    // Either the game ended naturally (length < max_moves) or hit the cap.
    assert!(examples.len() <= cfg.max_moves as usize);
}
