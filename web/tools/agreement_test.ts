// Numerical agreement test: run reference samples (produced by
// tools/gen_agreement_data.py) through the TS forward pass and assert
// the outputs match PyTorch's within a tight tolerance.
//
// Run from the web/ directory:
//     uv run python tools/gen_agreement_data.py     # produces web/tools/agreement_ref.json
//     cd web && npm run agreement-test
//
// The threshold is 1e-4 (max abs diff on policy logits, and on value).
// Float32 round-trip plus a dozen successive ops typically produces
// disagreement around 1e-6 to 1e-5; 1e-4 leaves headroom.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { Connect4 } from "../src/connect4";
import { Model, type Manifest } from "../src/model";

const __dirname = dirname(fileURLToPath(import.meta.url));

interface Sample {
  state: number[];
  player: number;
  n_moves_played: number;
  policy_logits: number[];
  value: number;
}

interface RefData {
  source_checkpoint: string;
  model_args: { num_res_blocks: number; num_hidden: number };
  samples: Sample[];
}

const TOLERANCE = 1e-4;

function loadManifestAndBlob(): { manifest: Manifest; buffer: ArrayBuffer } {
  const publicDir = resolve(__dirname, "..", "public");
  const manifest = JSON.parse(readFileSync(resolve(publicDir, "weights.json"), "utf-8")) as Manifest;
  const blob = readFileSync(resolve(publicDir, "weights.bin"));
  // Important: pass an ArrayBuffer view that exactly covers the file,
  // not the larger underlying Buffer that Node may have over-allocated.
  const buffer = blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.byteLength);
  return { manifest, buffer };
}

function loadRefData(): RefData {
  const path = resolve(__dirname, "agreement_ref.json");
  return JSON.parse(readFileSync(path, "utf-8")) as RefData;
}

function maxAbs(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

function main() {
  const { manifest, buffer } = loadManifestAndBlob();
  const ref = loadRefData();
  console.log(`Reference: ${ref.samples.length} samples from ${ref.source_checkpoint}`);
  console.log(`Manifest:  ${manifest.tensors.length} tensors, ${(buffer.byteLength / 1024).toFixed(1)} KB`);

  const game = new Connect4();
  const model = new Model(manifest, buffer);

  let maxLogitDiff = 0;
  let maxValueDiff = 0;
  let worstSampleIdx = -1;

  for (let i = 0; i < ref.samples.length; i++) {
    const s = ref.samples[i];
    const board = new Int8Array(s.state);
    const enc = game.encodeState(board, s.player);
    const result = model.forward(enc, false);
    const logitDiff = maxAbs(result.policyLogits, s.policy_logits);
    const valueDiff = Math.abs(result.value - s.value);
    if (logitDiff > maxLogitDiff || valueDiff > maxValueDiff) worstSampleIdx = i;
    if (logitDiff > maxLogitDiff) maxLogitDiff = logitDiff;
    if (valueDiff > maxValueDiff) maxValueDiff = valueDiff;
  }

  console.log(`Max abs diff:  policy_logits = ${maxLogitDiff.toExponential(3)}   value = ${maxValueDiff.toExponential(3)}`);
  console.log(`Worst sample index: ${worstSampleIdx}`);

  if (maxLogitDiff > TOLERANCE || maxValueDiff > TOLERANCE) {
    console.error(`FAIL: tolerance ${TOLERANCE.toExponential(0)} exceeded`);
    process.exit(1);
  }
  console.log(`PASS — within tolerance ${TOLERANCE.toExponential(0)}`);
}

main();
