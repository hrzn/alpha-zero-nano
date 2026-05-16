//! `selfplay` binary — entry point for the Phase 5 Python driver.
//!
//! Spawns N worker threads, each generating self-play games with MCTS +
//! Dirichlet noise + temperature sampling, sharing one `OnnxEvaluator`
//! behind a Mutex (so NN inference calls serialize while everything else
//! runs in parallel). When all games finish, the main thread writes three
//! `.npy` shards (`states.npy`, `policies.npy`, `values.npy`) into the
//! output directory.
//!
//! Args mirror the relevant fields of the Python preset dict so the
//! Phase 5 driver can pass them through verbatim. Example:
//!
//!     cargo run --release --bin selfplay -- \
//!         --model checkpoints/chess/model.onnx \
//!         --out shards/iter_001 \
//!         --num-games 40 --num-workers 4 \
//!         --num-searches 200 --c-puct 1.5 \
//!         --batch-size 32 --max-moves 200 --temp-threshold 30 \
//!         --dirichlet-alpha 0.3 --dirichlet-epsilon 0.25 \
//!         --seed 42

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use alpha_zero_nano::inference::OnnxEvaluator;
use alpha_zero_nano::selfplay::{play_game, Example, SelfPlayConfig};
use alpha_zero_nano::shards::write_shards;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Debug)]
struct Args {
    model: PathBuf,
    out: PathBuf,
    num_games: u32,
    num_workers: u32,
    cfg: SelfPlayConfig,
    seed: u64,
}

fn print_usage() {
    eprintln!(
        "selfplay --model <onnx> --out <dir> --num-games N --num-workers N \\\n\
         \t--num-searches N --c-puct F --batch-size N --max-moves N --temp-threshold N \\\n\
         \t--dirichlet-alpha F --dirichlet-epsilon F --seed N"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut num_games: u32 = 1;
    let mut num_workers: u32 = 1;
    let mut num_searches: u32 = 100;
    let mut c_puct: f64 = 1.0;
    let mut batch_size: u32 = 1;
    let mut max_moves: u32 = 200;
    let mut temp_threshold: Option<u32> = None;
    let mut dirichlet_alpha: f32 = 0.3;
    let mut dirichlet_epsilon: f32 = 0.25;
    let mut seed: u64 = 0;

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = &argv[i];
        let val = || -> Result<&str, String> {
            argv.get(i + 1)
                .map(|s| s.as_str())
                .ok_or_else(|| format!("missing value for {key}"))
        };
        match key.as_str() {
            "--model" => { model = Some(val()?.into()); i += 2; }
            "--out" => { out = Some(val()?.into()); i += 2; }
            "--num-games" => { num_games = val()?.parse().map_err(|e| format!("{e}"))?; i += 2; }
            "--num-workers" => { num_workers = val()?.parse().map_err(|e| format!("{e}"))?; i += 2; }
            "--num-searches" => { num_searches = val()?.parse().map_err(|e| format!("{e}"))?; i += 2; }
            "--c-puct" => { c_puct = val()?.parse().map_err(|e| format!("{e}"))?; i += 2; }
            "--batch-size" => { batch_size = val()?.parse().map_err(|e| format!("{e}"))?; i += 2; }
            "--max-moves" => { max_moves = val()?.parse().map_err(|e| format!("{e}"))?; i += 2; }
            "--temp-threshold" => {
                temp_threshold = Some(val()?.parse().map_err(|e| format!("{e}"))?);
                i += 2;
            }
            "--dirichlet-alpha" => { dirichlet_alpha = val()?.parse().map_err(|e| format!("{e}"))?; i += 2; }
            "--dirichlet-epsilon" => { dirichlet_epsilon = val()?.parse().map_err(|e| format!("{e}"))?; i += 2; }
            "--seed" => { seed = val()?.parse().map_err(|e| format!("{e}"))?; i += 2; }
            "-h" | "--help" => { print_usage(); std::process::exit(0); }
            unknown => return Err(format!("unknown arg: {unknown}")),
        }
    }

    Ok(Args {
        model: model.ok_or("--model is required")?,
        out: out.ok_or("--out is required")?,
        num_games,
        num_workers: num_workers.max(1),
        cfg: SelfPlayConfig {
            num_searches,
            c_puct,
            batch_size,
            dirichlet_alpha,
            dirichlet_epsilon,
            max_moves,
            temp_threshold,
        },
        seed,
    })
}

/// load-dynamic: discover the libonnxruntime dylib from the project's Python
/// virtualenv and set `ORT_DYLIB_PATH` before any ort call. Mirrors the
/// strategy used in the integration tests; production deployments will set
/// `ORT_DYLIB_PATH` directly via the wrapping driver.
fn ensure_ort_dylib() {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return;
    }
    // Walk up from the binary location to find a .venv/.
    let mut cur = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    while let Some(d) = cur {
        let candidate = d.join(".venv/lib/python3.12/site-packages/onnxruntime/capi");
        if candidate.exists() {
            if let Ok(entries) = std::fs::read_dir(&candidate) {
                for e in entries.flatten() {
                    let name = e.file_name();
                    let s = name.to_string_lossy();
                    if s.starts_with("libonnxruntime") && s.ends_with(".dylib") {
                        // Safety: single-threaded init phase before any ort call.
                        unsafe { std::env::set_var("ORT_DYLIB_PATH", e.path()); }
                        return;
                    }
                }
            }
        }
        cur = d.parent().map(Path::to_path_buf);
    }
    eprintln!(
        "warning: ORT_DYLIB_PATH not set and no .venv found — ort will fail to load"
    );
}

fn run() -> Result<(), String> {
    let args = parse_args().map_err(|e| {
        print_usage();
        e
    })?;

    ensure_ort_dylib();
    std::fs::create_dir_all(&args.out).map_err(|e| format!("mkdir {}: {e}", args.out.display()))?;

    let evaluator = Arc::new(
        OnnxEvaluator::new(&args.model)
            .map_err(|e| format!("load ONNX {}: {e}", args.model.display()))?,
    );

    // Distribute games across workers as evenly as we can.
    let n_workers = args.num_workers.min(args.num_games).max(1);
    let mut games_per_worker = vec![args.num_games / n_workers; n_workers as usize];
    for i in 0..(args.num_games % n_workers) as usize {
        games_per_worker[i] += 1;
    }

    println!(
        "selfplay: {} games across {} worker(s), seed={}, sims/move={}, batch={}",
        args.num_games, n_workers, args.seed, args.cfg.num_searches, args.cfg.batch_size,
    );
    let t0 = Instant::now();

    let mut handles = Vec::new();
    for (worker_id, games) in games_per_worker.iter().enumerate() {
        let ev = Arc::clone(&evaluator);
        let cfg = args.cfg.clone();
        let games = *games;
        let worker_seed = args.seed.wrapping_add(worker_id as u64);
        handles.push(thread::spawn(move || -> Vec<Example> {
            let mut rng = ChaCha8Rng::seed_from_u64(worker_seed);
            let mut out = Vec::new();
            for _ in 0..games {
                let examples = play_game(&*ev, &cfg, &mut rng);
                out.extend(examples);
            }
            out
        }));
    }

    let mut all: Vec<Example> = Vec::new();
    for h in handles {
        let chunk = h.join().map_err(|e| format!("worker panicked: {e:?}"))?;
        all.extend(chunk);
    }

    let elapsed = t0.elapsed();
    println!(
        "selfplay: produced {} examples in {:.1}s ({:.1} examples/s, {:.2}s/game avg)",
        all.len(),
        elapsed.as_secs_f64(),
        all.len() as f64 / elapsed.as_secs_f64().max(1e-6),
        elapsed.as_secs_f64() / args.num_games as f64,
    );

    if all.is_empty() {
        return Err("no examples produced — refusing to write empty shards".into());
    }

    write_shards(&args.out, &all).map_err(|e| format!("write_shards: {e}"))?;
    println!("selfplay: wrote shards to {}", args.out.display());
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
