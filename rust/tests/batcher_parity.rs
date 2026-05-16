//! Verify `BatchedOnnxEvaluator` returns the same outputs as `OnnxEvaluator`
//! on the same inputs (correctness), and is safe under concurrent load
//! (multiple threads hammering it return correct distinct results).

use std::path::PathBuf;
use std::sync::{Arc, Once};
use std::thread;
use std::time::Duration;

use alpha_zero_nano::game::GameState;
use alpha_zero_nano::inference::{BatchedOnnxEvaluator, BatcherConfig, Evaluator, OnnxEvaluator};

static ORT_INIT: Once = Once::new();

fn ensure_ort_dylib() {
    ORT_INIT.call_once(|| {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("project root")
            .to_path_buf();
        let venv_lib =
            project_root.join(".venv/lib/python3.12/site-packages/onnxruntime/capi");
        let dylib = std::fs::read_dir(&venv_lib)
            .ok()
            .and_then(|mut it| {
                it.find_map(|e| {
                    let e = e.ok()?;
                    let s = e.file_name().to_string_lossy().into_owned();
                    if s.starts_with("libonnxruntime") && s.ends_with(".dylib") {
                        Some(e.path())
                    } else {
                        None
                    }
                })
            })
            .unwrap_or_else(|| panic!("libonnxruntime not found in {}", venv_lib.display()));
        unsafe {
            std::env::set_var("ORT_DYLIB_PATH", &dylib);
        }
    });
}

fn onnx_path() -> PathBuf {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/chess_inference.onnx");
    if !p.exists() {
        panic!(
            "ONNX fixture missing at {}\n\
             Regenerate via: uv run python tools/gen_inference_parity_fixtures.py",
            p.display()
        );
    }
    p
}

/// Build a handful of distinct chess positions to feed the evaluator with.
fn sample_positions() -> Vec<GameState> {
    let mut out = vec![GameState::initial()];
    for moves in [
        vec!["e2e4"],
        vec!["e2e4", "e7e5"],
        vec!["e2e4", "e7e5", "g1f3"],
        vec!["e2e4", "e7e5", "g1f3", "b8c6"],
        vec!["d2d4", "g8f6"],
        vec!["c2c4", "e7e5", "b1c3"],
    ] {
        let mut s = GameState::initial();
        for uci in moves {
            s.push_uci(uci).unwrap();
        }
        out.push(s);
    }
    out
}

fn close_enough(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn batched_matches_mutex_evaluator() {
    ensure_ort_dylib();
    let path = onnx_path();
    let reference = OnnxEvaluator::new(&path).expect("OnnxEvaluator");
    let batched = BatchedOnnxEvaluator::new(&path).expect("BatchedOnnxEvaluator");

    let states = sample_positions();
    for (i, gs) in states.iter().enumerate() {
        let player = gs.turn();
        let (r_policy, r_value) = reference.evaluate(&gs.pos, player);
        let (b_policy, b_value) = batched.evaluate(&gs.pos, player);
        // The same Session impl runs both — agreement should be exact, but
        // give a touch of float tolerance to be safe across runtimes.
        assert!(
            close_enough(&r_policy, &b_policy, 1e-6),
            "position {i}: policy mismatch",
        );
        assert!(
            (r_value - b_value).abs() <= 1e-6,
            "position {i}: value mismatch (mutex={r_value}, batched={b_value})",
        );
    }
}

#[test]
fn concurrent_requests_return_correct_distinct_results() {
    // Spawn N threads that each repeatedly evaluate one of the sample
    // positions and check the result against the reference (computed
    // single-threaded up front). The batcher must hand the right answer
    // back to the right requester — a bug where responses get crossed
    // shows up immediately as a mismatch.
    ensure_ort_dylib();
    let path = onnx_path();
    let reference = OnnxEvaluator::new(&path).expect("OnnxEvaluator");
    let states = sample_positions();
    let expected: Vec<(Vec<f32>, f32)> = states
        .iter()
        .map(|gs| reference.evaluate(&gs.pos, gs.turn()))
        .collect();

    let batched = Arc::new(
        BatchedOnnxEvaluator::with_config(
            &path,
            BatcherConfig {
                max_batch_size: 64,
                batch_timeout: Duration::from_millis(2),
            },
        )
        .expect("BatchedOnnxEvaluator"),
    );

    let n_threads = 4;
    let iters_per_thread = 50;
    let mut handles = Vec::new();
    for t in 0..n_threads {
        let states = states.clone();
        let expected = expected.clone();
        let batched = Arc::clone(&batched);
        handles.push(thread::spawn(move || {
            for k in 0..iters_per_thread {
                let i = (t * 31 + k) % states.len();
                let gs = &states[i];
                let (got_policy, got_value) = batched.evaluate(&gs.pos, gs.turn());
                let (ref_policy, ref_value) = &expected[i];
                assert!(
                    close_enough(&got_policy, ref_policy, 1e-5),
                    "thread {t} iter {k} pos {i}: policy mismatch",
                );
                assert!(
                    (got_value - ref_value).abs() <= 1e-5,
                    "thread {t} iter {k} pos {i}: value mismatch",
                );
            }
        }));
    }
    for h in handles {
        h.join().expect("worker panic");
    }
}

#[test]
fn evaluate_batch_returns_per_leaf_results_in_order() {
    ensure_ort_dylib();
    let path = onnx_path();
    let reference = OnnxEvaluator::new(&path).expect("OnnxEvaluator");
    let batched = BatchedOnnxEvaluator::new(&path).expect("BatchedOnnxEvaluator");

    let states = sample_positions();
    let leaves: Vec<_> = states.iter().map(|gs| (gs.pos.clone(), gs.turn())).collect();

    let got = batched.evaluate_batch(&leaves);
    assert_eq!(got.len(), leaves.len());
    for (i, (policy, value)) in got.iter().enumerate() {
        let (ref_p, ref_v) = reference.evaluate(&leaves[i].0, leaves[i].1);
        assert!(
            close_enough(policy, &ref_p, 1e-5),
            "leaf {i}: policy mismatch",
        );
        assert!((value - ref_v).abs() <= 1e-5, "leaf {i}: value mismatch");
    }
}
