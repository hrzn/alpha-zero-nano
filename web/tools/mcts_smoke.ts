// Smoke test the full MCTS + model + game chain in Node, end-to-end.
// Runs 100 sims on the initial Connect 4 position and prints the resulting
// visit-count distribution. Catches port bugs that the model-only
// agreement test doesn't see (e.g., wrong sign convention, expand bug).

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { Connect4 } from "../src/connect4";
import { Model, type Manifest } from "../src/model";
import { MCTS } from "../src/mcts";

const __dirname = dirname(fileURLToPath(import.meta.url));

const publicDir = resolve(__dirname, "..", "public");
const manifest = JSON.parse(readFileSync(resolve(publicDir, "weights.json"), "utf-8")) as Manifest;
const blob = readFileSync(resolve(publicDir, "weights.bin"));
const buffer = blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.byteLength);

const game = new Connect4();
const model = new Model(manifest, buffer);
const mcts = new MCTS(game, model, { numSearches: 100, cPuct: 1.0 });

const t0 = performance.now();
const probs = mcts.search(game.getInitialState(), 1);
const elapsed = performance.now() - t0;

console.log(`100 sims completed in ${elapsed.toFixed(0)} ms`);
console.log("Visit-count distribution over columns:");
let total = 0;
for (let c = 0; c < probs.length; c++) total += probs[c];
for (let c = 0; c < probs.length; c++) {
  const bar = "█".repeat(Math.round(probs[c] * 40));
  console.log(`  ${c}: ${(probs[c] * 100).toFixed(1).padStart(5)}%  ${bar}`);
}
console.log(`(probs sum = ${total.toFixed(4)} — should be 1.0)`);
console.log(`argmax column = ${probs.indexOf(Math.max(...probs))}`);
