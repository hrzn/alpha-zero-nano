//! Phase 3 inference parity: Rust ort vs PyTorch eager forward.
//!
//! Loads the ONNX model + reference data emitted by
//! ``tools/gen_inference_parity_fixtures.py``. For each random `(17, 8, 8)`
//! input, runs `OnnxEvaluator::forward` and asserts the output matches
//! PyTorch's eager forward within 1e-4 on policy logits and value — the
//! same tolerance the Phase 0 onnxruntime-python test uses
//! (`tools/test_export_chess_onnx.py`).
//!
//! Fixture files are large (~33 MB ONNX, ~2 MB JSON) and live in
//! `tests/fixtures/`; both are git-ignored and regenerated via
//!     uv run python tools/gen_inference_parity_fixtures.py
//! The test panics with a clear "regenerate" message if they're missing.

use std::fs;
use std::path::PathBuf;
use std::sync::Once;

use alpha_zero_nano::inference::OnnxEvaluator;
use ndarray::Array4;
use serde::Deserialize;

const TOL: f32 = 1e-4;

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u32,
    input_shape: [usize; 3],
    samples: Vec<Sample>,
}

#[derive(Debug, Deserialize)]
struct Sample {
    input: Vec<f32>,
    expected_policy_logits: Vec<f32>,
    expected_value: f32,
}

static ORT_INIT: Once = Once::new();

/// Locate the libonnxruntime dylib shipped with the project's Python
/// `onnxruntime` package. We use load-dynamic to avoid the ort build-time
/// download (which transitively requires edition2024 deps awkward to pin).
fn ensure_ort_dylib() {
    ORT_INIT.call_once(|| {
        // Project root: ../ relative to this file.
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let project_root = manifest_dir.parent().expect("project root");
        let venv_lib = project_root
            .join(".venv")
            .join("lib")
            .join("python3.12")
            .join("site-packages")
            .join("onnxruntime")
            .join("capi");
        let dylib = std::fs::read_dir(&venv_lib)
            .ok()
            .and_then(|mut it| {
                it.find_map(|e| {
                    let e = e.ok()?;
                    let name = e.file_name();
                    let s = name.to_string_lossy();
                    if s.starts_with("libonnxruntime") && s.ends_with(".dylib") {
                        Some(e.path())
                    } else {
                        None
                    }
                })
            })
            .unwrap_or_else(|| {
                panic!(
                    "could not find libonnxruntime in {}; install via `uv sync --group dev`",
                    venv_lib.display()
                )
            });
        // Safety: setting env vars before any other thread reads them — only
        // OnnxEvaluator::new will, and it runs after this Once block.
        unsafe {
            std::env::set_var("ORT_DYLIB_PATH", &dylib);
        }
    });
}

fn fixture_paths() -> (PathBuf, PathBuf) {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    (
        base.join("chess_inference.onnx"),
        base.join("chess_inference.json"),
    )
}

fn load_fixture() -> (PathBuf, Fixture) {
    let (onnx_path, json_path) = fixture_paths();
    if !onnx_path.exists() || !json_path.exists() {
        panic!(
            "inference fixtures missing.\n\
             Expected:\n  {}\n  {}\n\
             Regenerate via:\n  uv run python tools/gen_inference_parity_fixtures.py",
            onnx_path.display(),
            json_path.display(),
        );
    }
    let text = fs::read_to_string(&json_path).expect("read fixture json");
    let f: Fixture = serde_json::from_str(&text).expect("parse fixture json");
    assert_eq!(f.schema_version, 1);
    assert_eq!(f.input_shape, [17, 8, 8]);
    (onnx_path, f)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn ort_matches_pytorch_eager_per_sample() {
    ensure_ort_dylib();
    let (onnx_path, fixture) = load_fixture();
    let evaluator = OnnxEvaluator::new(&onnx_path).expect("create OnnxEvaluator");

    for (i, sample) in fixture.samples.iter().enumerate() {
        let arr =
            Array4::from_shape_vec((1, 17, 8, 8), sample.input.clone()).expect("input shape");
        let (logits, values) = evaluator.forward(&arr).expect("ort forward");
        assert_eq!(logits.len(), 1);
        assert_eq!(values.len(), 1);
        assert_eq!(logits[0].len(), 4096);

        let logit_diff = max_abs_diff(&logits[0], &sample.expected_policy_logits);
        assert!(
            logit_diff < TOL,
            "sample {i}: policy_logits max abs diff {logit_diff} >= {TOL}",
        );
        let value_diff = (values[0] - sample.expected_value).abs();
        assert!(
            value_diff < TOL,
            "sample {i}: value diff {value_diff} >= {TOL}  (got {}, want {})",
            values[0],
            sample.expected_value,
        );
    }
}

#[test]
fn ort_matches_pytorch_eager_batched() {
    // Run all 16 samples in one batch — same outputs as per-sample.
    ensure_ort_dylib();
    let (onnx_path, fixture) = load_fixture();
    let evaluator = OnnxEvaluator::new(&onnx_path).expect("create OnnxEvaluator");

    let n = fixture.samples.len();
    let mut buf = Vec::with_capacity(n * 17 * 8 * 8);
    for s in &fixture.samples {
        buf.extend_from_slice(&s.input);
    }
    let arr = Array4::from_shape_vec((n, 17, 8, 8), buf).expect("batch shape");
    let (logits, values) = evaluator.forward(&arr).expect("ort batch forward");
    assert_eq!(logits.len(), n);
    assert_eq!(values.len(), n);

    for (i, sample) in fixture.samples.iter().enumerate() {
        let logit_diff = max_abs_diff(&logits[i], &sample.expected_policy_logits);
        assert!(
            logit_diff < TOL,
            "batched sample {i}: policy_logits max abs diff {logit_diff} >= {TOL}",
        );
        let value_diff = (values[i] - sample.expected_value).abs();
        assert!(
            value_diff < TOL,
            "batched sample {i}: value diff {value_diff} >= {TOL}",
        );
    }
}
