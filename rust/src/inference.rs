//! Pluggable model evaluator used by MCTS to score leaves.
//!
//! `Evaluator` is the abstraction; concrete impls:
//!   - `FixtureEvaluator` — lookup table backed by a JSON fixture, for
//!     algorithm parity tests where the NN must not be a variable.
//!   - `OnnxEvaluator` — production path, loads an ONNX model via `ort`
//!     and runs forward in batched f32 inference.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array4, Axis};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use shakmaty::{fen::Fen, Chess, EnPassantMode};

use crate::action::ACTION_SIZE;
use crate::encoding::{encode_state, Player};

/// Returns `(policy, value)` where:
///   policy: probability over all 4096 actions, length == ACTION_SIZE.
///           MCTS masks illegal indices and renormalises — the evaluator
///           does not need to.
///   value:  in [-1, 1], from `player`'s perspective.
pub trait Evaluator {
    fn evaluate(&self, pos: &Chess, player: Player) -> (Vec<f32>, f32);

    /// Batched variant. Default impl falls back to per-leaf calls; impls
    /// backed by a real NN should override this.
    fn evaluate_batch(&self, leaves: &[(Chess, Player)]) -> Vec<(Vec<f32>, f32)> {
        leaves
            .iter()
            .map(|(p, pl)| self.evaluate(p, *pl))
            .collect()
    }
}

/// Lookup-table evaluator backed by the JSON fixture produced by
/// `tools/gen_mcts_parity_fixtures.py`. Keys are `"<fen>|<player_int>"`.
///
/// Stored in sparse form (only legal-action priors). Returns a dense 4096
/// vector with priors at the stored indices and 0 everywhere else — the
/// same shape Python MCTS sees post-`policy * valid_moves`.
pub struct FixtureEvaluator {
    pub by_key: HashMap<String, FixtureEntry>,
}

#[derive(Debug, Clone)]
pub struct FixtureEntry {
    pub policy_actions: Vec<u16>,
    pub policy_priors: Vec<f32>,
    pub value: f32,
}

impl FixtureEvaluator {
    pub fn new(by_key: HashMap<String, FixtureEntry>) -> Self {
        Self { by_key }
    }

    fn key(pos: &Chess, player: Player) -> String {
        let fen = Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string();
        let p_int = match player {
            Player::White => 1,
            Player::Black => -1,
        };
        format!("{fen}|{p_int}")
    }
}

impl Evaluator for FixtureEvaluator {
    fn evaluate(&self, pos: &Chess, player: Player) -> (Vec<f32>, f32) {
        let key = Self::key(pos, player);
        let entry = self
            .by_key
            .get(&key)
            .unwrap_or_else(|| panic!("fixture miss for {key} — MCTS diverged from Python"));
        let mut policy = vec![0.0f32; ACTION_SIZE];
        for (&a, &p) in entry.policy_actions.iter().zip(entry.policy_priors.iter()) {
            policy[a as usize] = p;
        }
        (policy, entry.value)
    }
}

/// Test/dev evaluator returning a uniform policy and `value = 0` for every
/// position. Useful when wiring up self-play code paths without depending on
/// a trained model — exercises MCTS expansion, Dirichlet noise, temperature
/// sampling, and the shard writer without any NN dependency.
#[derive(Debug, Default, Clone, Copy)]
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate(&self, _pos: &Chess, _player: Player) -> (Vec<f32>, f32) {
        let p = 1.0 / ACTION_SIZE as f32;
        (vec![p; ACTION_SIZE], 0.0)
    }
}

/// Production evaluator: loads a chess ResNet exported to ONNX (via
/// `tools/export_chess_onnx.py`) and runs forward in f32 through `ort`.
///
/// Inputs are `(B, 17, 8, 8)` produced by the project's `encode_state`;
/// outputs are `(B, 4096)` policy logits (we softmax them) and `(B, 1)`
/// value in `[-1, 1]` from the side-to-move's perspective. Matches the
/// Python model interface at `model/model.py:62-69`.
///
/// **load-dynamic note:** ort here uses the `load-dynamic` feature, so the
/// libonnxruntime dylib must be reachable. Set `ORT_DYLIB_PATH` to the
/// .dylib that ships with the project's `onnxruntime` Python package
/// (`.venv/lib/.../onnxruntime/capi/libonnxruntime.*.dylib`) before
/// constructing an `OnnxEvaluator`.
pub struct OnnxEvaluator {
    // `ort::Session::run` requires `&mut self` even though the C++ ORT
    // runtime is thread-safe for Run() calls. We wrap in a Mutex to keep
    // the `Evaluator` trait at `&self`. Phase 4 (parallel self-play) may
    // revisit this — a shared batcher thread is the standard pattern.
    session: Mutex<Session>,
}

impl OnnxEvaluator {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, ort::Error> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path.as_ref())?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Run a batched forward on already-encoded inputs `(B, 17, 8, 8)`.
    /// Returns (logits[B, 4096], values[B]).
    pub fn forward(&self, batch: &Array4<f32>) -> Result<(Vec<Vec<f32>>, Vec<f32>), ort::Error> {
        let input_value = Value::from_array(batch.clone())?;
        let mut session = self.session.lock().expect("session mutex poisoned");
        let outputs = session.run(ort::inputs!["state" => input_value])?;

        let logits = outputs["policy_logits"].try_extract_array::<f32>()?;
        let values = outputs["value"].try_extract_array::<f32>()?;

        let batch_size = batch.shape()[0];
        let logits_view = logits.view();
        let values_view = values.view();

        let mut logits_out = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let row: Vec<f32> = logits_view
                .index_axis(Axis(0), i)
                .iter()
                .copied()
                .collect();
            logits_out.push(row);
        }

        let values_out: Vec<f32> = (0..batch_size)
            .map(|i| values_view.index_axis(Axis(0), i).iter().copied().next().unwrap())
            .collect();

        Ok((logits_out, values_out))
    }
}

/// Numerically-stable softmax over a logit vector. Mirrors
/// `torch.softmax(logits, dim=1)` which is what `model.predict` applies
/// before MCTS sees the policy.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in exps.iter_mut() {
            *v *= inv;
        }
    }
    exps
}

impl Evaluator for OnnxEvaluator {
    fn evaluate(&self, pos: &Chess, player: Player) -> (Vec<f32>, f32) {
        let enc = encode_state(pos, player);
        // (1, 17, 8, 8) for single-leaf path.
        let arr = Array4::from_shape_vec((1, 17, 8, 8), enc.0).expect("encode shape");
        let (logits, values) = self
            .forward(&arr)
            .expect("ort forward failed in single evaluate");
        (softmax(&logits[0]), values[0])
    }

    fn evaluate_batch(&self, leaves: &[(Chess, Player)]) -> Vec<(Vec<f32>, f32)> {
        if leaves.is_empty() {
            return Vec::new();
        }
        let batch_size = leaves.len();
        // Pack encoded inputs into one (B, 17, 8, 8) tensor.
        let mut buf = Vec::with_capacity(batch_size * 17 * 8 * 8);
        for (pos, player) in leaves {
            let enc = encode_state(pos, *player);
            buf.extend_from_slice(&enc.0);
        }
        let arr = Array4::from_shape_vec((batch_size, 17, 8, 8), buf).expect("batch shape");
        let (logits, values) = self.forward(&arr).expect("ort forward failed in batch");
        logits
            .into_iter()
            .zip(values.into_iter())
            .map(|(l, v)| (softmax(&l), v))
            .collect()
    }
}
