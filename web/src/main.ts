// Top-level wiring: load the model once, render the board, manage the
// human-vs-agent turn loop, and update the side panel after each agent move.

import { Connect4 } from "./connect4";
import { loadModel, type Model } from "./model";
import { MCTS } from "./mcts";

const game = new Connect4();
let state = game.getInitialState();
let humanPlayer = 1;
let currentPlayer = 1;
let model: Model | null = null;
let mcts: MCTS | null = null;
let busy = false;
let gameOver = false;
let winningCells: number[] = []; // flat indices

const boardEl = document.getElementById("board")!;
const statusEl = document.getElementById("status")!;
const resetBtn = document.getElementById("reset") as HTMLButtonElement;
const simsInput = document.getElementById("sims") as HTMLInputElement;
const simsReadout = document.getElementById("sims-readout")!;
const policyBarsEl = document.getElementById("policy-bars")!;
const valueReadoutEl = document.getElementById("value-readout")!;

function setStatus(msg: string) {
  statusEl.textContent = msg;
}

function findWinningCells(s: Int8Array, lastAction: number): number[] {
  // Recompute the 4 in a row through the topmost piece in the last-played column.
  let row = -1;
  for (let r = 0; r < game.rowCount; r++) {
    if (s[r * game.columnCount + lastAction] !== 0) { row = r; break; }
  }
  if (row === -1) return [];
  const player = s[row * game.columnCount + lastAction];
  const dirs: [number, number][] = [[0, 1], [1, 0], [1, 1], [1, -1]];
  for (const [dr, dc] of dirs) {
    const cells = [row * game.columnCount + lastAction];
    let r = row + dr, c = lastAction + dc;
    while (r >= 0 && r < game.rowCount && c >= 0 && c < game.columnCount && s[r * game.columnCount + c] === player) {
      cells.push(r * game.columnCount + c); r += dr; c += dc;
    }
    r = row - dr; c = lastAction - dc;
    while (r >= 0 && r < game.rowCount && c >= 0 && c < game.columnCount && s[r * game.columnCount + c] === player) {
      cells.push(r * game.columnCount + c); r -= dr; c -= dc;
    }
    if (cells.length >= 4) return cells;
  }
  return [];
}

function renderBoard() {
  // Rebuild children only when needed, but for simplicity we just rebuild.
  // 42 cells, cheap.
  boardEl.innerHTML = "";
  const valid = game.getValidMoves(state);
  for (let r = 0; r < game.rowCount; r++) {
    for (let c = 0; c < game.columnCount; c++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      const v = state[r * game.columnCount + c];
      if (v === humanPlayer) cell.classList.add("p1");
      else if (v === -humanPlayer) cell.classList.add("p2");
      if (winningCells.includes(r * game.columnCount + c)) cell.classList.add("win");

      const canPlay = !busy && !gameOver && currentPlayer === humanPlayer && valid[c] === 1;
      if (canPlay) {
        cell.classList.add("hint");
        cell.addEventListener("click", () => onHumanDrop(c));
      }
      boardEl.appendChild(cell);
    }
  }
}

function renderPolicy(
  probs: Float32Array | null,
  priors: Float32Array | null,
  valid: Uint8Array | null,
) {
  policyBarsEl.innerHTML = "";
  // Shared scale so visit-count bars and prior markers are visually comparable.
  let scaleMax = 0;
  if (probs) for (let i = 0; i < probs.length; i++) if (probs[i] > scaleMax) scaleMax = probs[i];
  if (priors) for (let i = 0; i < priors.length; i++) if (priors[i] > scaleMax) scaleMax = priors[i];

  let argmax = -1;
  if (probs) {
    let best = -1;
    for (let i = 0; i < probs.length; i++) if (probs[i] > best) { best = probs[i]; argmax = i; }
  }

  for (let c = 0; c < game.columnCount; c++) {
    const bar = document.createElement("div");
    bar.className = "bar";
    const isValid = !valid || valid[c] === 1;
    if (!isValid) bar.classList.add("dimmed");
    if (c === argmax && isValid) bar.classList.add("argmax");

    const p = probs ? probs[c] : 0;
    const prior = priors ? priors[c] : 0;

    // Filled bar = MCTS visit-count probability.
    const fill = document.createElement("div");
    fill.className = "fill";
    const heightPct = scaleMax > 0 ? (p / scaleMax) * 100 : 0;
    fill.style.height = `${heightPct}%`;
    bar.appendChild(fill);

    // Thin horizontal line = network prior probability for this column.
    if (priors && prior > 0) {
      const priorMark = document.createElement("div");
      priorMark.className = "prior";
      const priorPct = scaleMax > 0 ? (prior / scaleMax) * 100 : 0;
      priorMark.style.bottom = `${priorPct}%`;
      bar.appendChild(priorMark);
    }

    // Percentage label above the bar.
    const pct = document.createElement("div");
    pct.className = "pct";
    if (probs) pct.textContent = `${Math.round(p * 100)}%`;
    bar.appendChild(pct);

    // Column index below.
    const lbl = document.createElement("div");
    lbl.className = "col-label";
    lbl.textContent = String(c);
    bar.appendChild(lbl);

    policyBarsEl.appendChild(bar);
  }
}

function renderValue(value: number | null) {
  if (value === null) { valueReadoutEl.textContent = "—"; valueReadoutEl.style.color = ""; return; }
  const sign = value >= 0 ? "+" : "";
  valueReadoutEl.textContent = `${sign}${value.toFixed(2)}`;
  valueReadoutEl.style.color = value > 0.05 ? "var(--accent)" : value < -0.05 ? "#ff7777" : "var(--fg-dim)";
}

function applyMove(action: number, by: number): { terminated: boolean; value: number } {
  state = game.updateState(state, action, by);
  mcts!.advanceRoot(action);
  return game.getValueAndTerminated(state, action);
}

async function onHumanDrop(col: number) {
  if (busy || gameOver || currentPlayer !== humanPlayer) return;
  const valid = game.getValidMoves(state);
  if (!valid[col]) return;
  busy = true;
  const { terminated, value } = applyMove(col, currentPlayer);
  if (terminated) {
    winningCells = value === 1 ? findWinningCells(state, col) : [];
    gameOver = true;
    renderBoard();
    setStatus(value === 1 ? "You win." : "Draw.");
    busy = false;
    return;
  }
  currentPlayer = game.getOpponent(currentPlayer);
  renderBoard();
  setStatus("agent thinking…");

  // Yield to the event loop so the browser repaints "thinking…" before the
  // synchronous MCTS hot loop locks the main thread.
  await new Promise((r) => setTimeout(r, 16));

  const t0 = performance.now();
  const probs = mcts!.search(state, currentPlayer);
  const elapsed = performance.now() - t0;

  // Pick action greedily — highest visit count.
  let bestAction = -1;
  let bestProb = -1;
  for (let i = 0; i < probs.length; i++) if (probs[i] > bestProb) { bestProb = probs[i]; bestAction = i; }

  const root = mcts!.getRoot();
  const rootValue = root && root.visitCount > 0 ? -root.valueSum / root.visitCount : null;
  // Pull the network's raw priors per column straight off the root's children —
  // these are the post-softmax, valid-move-masked priors that MCTS expanded with.
  let priors: Float32Array | null = null;
  if (root) {
    priors = new Float32Array(game.actionSize);
    for (const [action, child] of root.children) priors[action] = child.prior;
  }
  renderPolicy(probs, priors, game.getValidMoves(state));
  renderValue(rootValue);

  const { terminated: t2, value: v2 } = applyMove(bestAction, currentPlayer);
  setStatus(`agent played col ${bestAction}  (${elapsed.toFixed(0)} ms)`);

  if (t2) {
    winningCells = v2 === 1 ? findWinningCells(state, bestAction) : [];
    gameOver = true;
    renderBoard();
    setStatus(v2 === 1 ? "Agent wins." : "Draw.");
    busy = false;
    return;
  }
  currentPlayer = game.getOpponent(currentPlayer);
  // Important: clear `busy` BEFORE rendering, otherwise renderBoard's
  // canPlay check (`!busy && ...`) falsifies and no click handlers attach.
  busy = false;
  renderBoard();
  setStatus(`your move  (agent thought for ${elapsed.toFixed(0)} ms)`);
}

function reset() {
  state = game.getInitialState();
  currentPlayer = 1;
  humanPlayer = 1;
  busy = false;
  gameOver = false;
  winningCells = [];
  if (mcts) mcts.resetRoot();
  renderBoard();
  renderPolicy(null, null, null);
  renderValue(null);
  setStatus("your move");
}

resetBtn.addEventListener("click", reset);
simsInput.addEventListener("input", () => {
  const n = parseInt(simsInput.value, 10);
  simsReadout.textContent = String(n);
  if (mcts) mcts.opts.numSearches = n;
});

async function init() {
  setStatus("loading model…");
  try {
    // Relative paths work on both the dev server and any GitHub Pages subpath
    // (e.g. https://user.github.io/alpha-zero-nano/) without further config.
    model = await loadModel("weights.json", "weights.bin");
  } catch (err) {
    setStatus(`load failed: ${(err as Error).message}`);
    throw err;
  }
  const sims = parseInt(simsInput.value, 10);
  simsReadout.textContent = String(sims);
  mcts = new MCTS(game, model, { numSearches: sims, cPuct: 1.0 });
  reset();
  setStatus("your move — drop a piece in any column");
}

init();
