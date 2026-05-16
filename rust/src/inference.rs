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
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
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

// ── Batched ONNX evaluator with a dedicated batcher thread ──────────────────
//
// `OnnxEvaluator`'s Mutex<Session> serialises one Session::run per call. With
// N self-play workers each doing batched MCTS at internal batch=B, the four
// workers' B-sized inference calls execute one after another rather than
// being combined into one (4×B)-sized call. That's the Phase 4 deferred
// optimisation; this struct fixes it.
//
// Architecture: worker threads call `evaluate` / `evaluate_batch` on the
// shared `BatchedOnnxEvaluator`. Each request goes through a crossbeam
// channel to a single batcher thread that owns the `Session`. The batcher
// pulls requests, accumulates them up to `max_batch_size` or until
// `batch_timeout` elapses, runs ONE `Session::run`, dispatches per-row
// responses back through per-request one-shot channels.

#[derive(Debug, Clone, Copy)]
pub struct BatcherConfig {
    /// Hard cap on how many leaves the batcher will pack into one inference
    /// call. Bigger = better hardware utilisation but more memory. With 4
    /// workers at MCTS batch=32, expect bursts of ~128; default leaves
    /// headroom for 8 workers.
    pub max_batch_size: usize,
    /// After the first request lands, the batcher waits up to this long for
    /// more requests to coalesce. Trade-off: longer = bigger batches but
    /// higher per-call latency. CPU inference is throughput-bound so a few
    /// ms is fine.
    pub batch_timeout: Duration,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            batch_timeout: Duration::from_millis(2),
        }
    }
}

struct InferenceRequest {
    state: Chess,
    player: Player,
    response: Sender<(Vec<f32>, f32)>,
}

pub struct BatchedOnnxEvaluator {
    tx: Sender<InferenceRequest>,
    // Hold the join handle to keep the thread parented to this evaluator —
    // when the evaluator drops, the channel closes, the batcher exits, and
    // the thread is detached cleanly.
    _batcher: Option<thread::JoinHandle<()>>,
}

impl BatchedOnnxEvaluator {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, ort::Error> {
        Self::with_config(model_path, BatcherConfig::default())
    }

    pub fn with_config(
        model_path: impl AsRef<Path>,
        cfg: BatcherConfig,
    ) -> Result<Self, ort::Error> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path.as_ref())?;
        let (tx, rx) = unbounded::<InferenceRequest>();
        let handle = thread::Builder::new()
            .name("ort-batcher".into())
            .spawn(move || run_batcher(session, rx, cfg))
            .expect("spawn batcher thread");
        Ok(Self {
            tx,
            _batcher: Some(handle),
        })
    }

    fn submit(&self, state: Chess, player: Player) -> Receiver<(Vec<f32>, f32)> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(InferenceRequest {
                state,
                player,
                response: resp_tx,
            })
            .expect("batcher thread is gone");
        resp_rx
    }
}

impl Drop for BatchedOnnxEvaluator {
    fn drop(&mut self) {
        // Drop the sender to signal the batcher to exit, then join so the
        // batcher's Session is destructed on its own thread (avoids any
        // potential thread-affinity issues in ort).
        // Note: we hold tx until here; close by replacing with a dummy.
        // Simpler: just don't join — let the OS clean up the detached thread
        // when the process exits, which is the normal shutdown path for the
        // selfplay binary.
        let _ = self._batcher.take();
    }
}

impl Evaluator for BatchedOnnxEvaluator {
    fn evaluate(&self, pos: &Chess, player: Player) -> (Vec<f32>, f32) {
        let rx = self.submit(pos.clone(), player);
        rx.recv().expect("batcher dropped before responding")
    }

    fn evaluate_batch(&self, leaves: &[(Chess, Player)]) -> Vec<(Vec<f32>, f32)> {
        if leaves.is_empty() {
            return Vec::new();
        }
        // Submit all at once so the batcher sees them as a burst and is more
        // likely to coalesce them with other workers' requests into one call.
        let receivers: Vec<_> = leaves
            .iter()
            .map(|(pos, player)| self.submit(pos.clone(), *player))
            .collect();
        receivers
            .into_iter()
            .map(|rx| rx.recv().expect("batcher dropped before responding"))
            .collect()
    }
}

fn run_batcher(mut session: Session, rx: Receiver<InferenceRequest>, cfg: BatcherConfig) {
    loop {
        // Block waiting for the first request of the next batch.
        let first = match rx.recv() {
            Ok(req) => req,
            Err(_) => return, // all senders dropped — clean exit
        };
        let mut batch: Vec<InferenceRequest> = Vec::with_capacity(cfg.max_batch_size);
        batch.push(first);

        // Top up the batch with whatever else arrives within the timeout
        // window, up to max_batch_size. With multiple worker threads doing
        // batched MCTS, ~32 leaves per worker round all land within
        // microseconds of each other — we want them in one inference call.
        let deadline = Instant::now() + cfg.batch_timeout;
        while batch.len() < cfg.max_batch_size {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                // Drain anything that's already queued without blocking.
                while batch.len() < cfg.max_batch_size {
                    match rx.try_recv() {
                        Ok(req) => batch.push(req),
                        Err(_) => break,
                    }
                }
                break;
            }
            match rx.recv_timeout(remaining) {
                Ok(req) => batch.push(req),
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => break,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    // Senders gone — finish this batch and exit.
                    if let Err(e) = run_one_batch(&mut session, &batch) {
                        eprintln!("batcher: final inference failed: {e}");
                    }
                    return;
                }
            }
        }

        if let Err(e) = run_one_batch(&mut session, &batch) {
            eprintln!("batcher: inference failed for batch of {}: {e}", batch.len());
            // Drop the requesters' Sender halves — they'll see RecvError.
        }
    }
}

fn run_one_batch(
    session: &mut Session,
    batch: &[InferenceRequest],
) -> Result<(), ort::Error> {
    let n = batch.len();
    let mut buf = Vec::with_capacity(n * 17 * 8 * 8);
    for req in batch {
        let enc = encode_state(&req.state, req.player);
        buf.extend_from_slice(&enc.0);
    }
    let arr = Array4::from_shape_vec((n, 17, 8, 8), buf).expect("batch shape");
    let input_value = Value::from_array(arr)?;
    let outputs = session.run(ort::inputs!["state" => input_value])?;

    let logits = outputs["policy_logits"].try_extract_array::<f32>()?;
    let values = outputs["value"].try_extract_array::<f32>()?;
    let logits_view = logits.view();
    let values_view = values.view();

    for (i, req) in batch.iter().enumerate() {
        let row: Vec<f32> = logits_view
            .index_axis(Axis(0), i)
            .iter()
            .copied()
            .collect();
        let value: f32 = values_view
            .index_axis(Axis(0), i)
            .iter()
            .copied()
            .next()
            .unwrap();
        let policy = softmax(&row);
        // If the requester dropped, just discard — it's not an error here.
        let _ = req.response.send((policy, value));
    }
    Ok(())
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
