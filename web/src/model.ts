// Hand-rolled forward pass for the trained C4 ResNet.
//
// Why hand-rolled (not ONNX Runtime Web): the educational page wants
// intermediate activations from every layer. Running in TS gives us
// every tensor as a local variable we can hand to the renderer, with
// no opaque WASM runtime in between. Performance is fine: the model is
// ~190K params on a 6x7 input, ~10ms per forward in pure JS on M1.
//
// Tensor layout: NCHW, batch dimension dropped (batch=1 always).
// A `Tensor` is a flat Float32Array plus a shape array.

export interface Tensor {
  data: Float32Array;
  shape: number[];
}

export interface ManifestTensor {
  name: string;
  shape: number[];
  offset: number;
  size: number;
}

export interface Manifest {
  tensors: ManifestTensor[];
  model_args: { num_res_blocks: number; num_hidden: number };
  game: { row_count: number; column_count: number; action_size: number; num_channels: number };
  bn_eps: number;
  iteration: number | null;
  source_checkpoint: string;
}

export interface ForwardResult {
  policyLogits: Float32Array; // [actionSize]
  policy: Float32Array;       // softmax of logits
  value: number;              // tanh-clipped scalar in [-1, 1]
  activations: Map<string, Tensor>;
}

class Weights {
  private byName: Map<string, ManifestTensor>;
  private buffer: ArrayBuffer;

  constructor(manifest: Manifest, buffer: ArrayBuffer) {
    this.byName = new Map(manifest.tensors.map((t) => [t.name, t]));
    this.buffer = buffer;
  }

  get(name: string): Float32Array {
    const meta = this.byName.get(name);
    if (!meta) throw new Error(`Tensor ${name} not in manifest`);
    return new Float32Array(this.buffer, meta.offset, meta.size / 4);
  }
}

// ── Ops ────────────────────────────────────────────────────────────────────

/**
 * 2D convolution with stride=1 and given padding.
 * - input:  shape [Cin, H, W]
 * - weight: flat NCHW [Cout, Cin, kH, kW]
 * - bias:   flat [Cout]
 * - output: shape [Cout, H, W]   (same H, W when padding = (kH-1)/2)
 */
function conv2d(
  input: Tensor,
  weight: Float32Array,
  bias: Float32Array,
  outChannels: number,
  inChannels: number,
  kH: number,
  kW: number,
  padding: number,
): Tensor {
  const [, H, W] = input.shape;
  const out = new Float32Array(outChannels * H * W);
  const inData = input.data;
  const HW = H * W;
  const kHkW = kH * kW;
  for (let oc = 0; oc < outChannels; oc++) {
    const ocBias = bias[oc];
    const wOcBase = oc * inChannels * kHkW;
    const outOcBase = oc * HW;
    for (let h = 0; h < H; h++) {
      // Precompute the valid kh range for this row, so the inner loop has no bounds checks.
      const khMin = h - padding < 0 ? padding - h : 0;
      const khMax = h - padding + kH > H ? H + padding - h : kH;
      for (let w = 0; w < W; w++) {
        const kwMin = w - padding < 0 ? padding - w : 0;
        const kwMax = w - padding + kW > W ? W + padding - w : kW;
        let sum = ocBias;
        for (let ic = 0; ic < inChannels; ic++) {
          const inIcBase = ic * HW;
          const wIcBase = wOcBase + ic * kHkW;
          for (let kh = khMin; kh < khMax; kh++) {
            const ih = h - padding + kh;
            const inRowBase = inIcBase + ih * W;
            const wRowBase = wIcBase + kh * kW;
            for (let kw = kwMin; kw < kwMax; kw++) {
              sum += inData[inRowBase + (w - padding + kw)] * weight[wRowBase + kw];
            }
          }
        }
        out[outOcBase + h * W + w] = sum;
      }
    }
  }
  return { data: out, shape: [outChannels, H, W] };
}

/**
 * BatchNorm in eval mode: per-channel affine using stored running stats.
 *   y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
 */
function bn(
  input: Tensor,
  weight: Float32Array,
  bias: Float32Array,
  runningMean: Float32Array,
  runningVar: Float32Array,
  eps: number,
): Tensor {
  const [C, H, W] = input.shape;
  const out = new Float32Array(input.data.length);
  const HW = H * W;
  for (let c = 0; c < C; c++) {
    const scale = weight[c] / Math.sqrt(runningVar[c] + eps);
    const shift = bias[c] - runningMean[c] * scale;
    const base = c * HW;
    for (let i = 0; i < HW; i++) {
      out[base + i] = input.data[base + i] * scale + shift;
    }
  }
  return { data: out, shape: input.shape };
}

function relu(input: Tensor): Tensor {
  const out = new Float32Array(input.data.length);
  for (let i = 0; i < input.data.length; i++) out[i] = input.data[i] > 0 ? input.data[i] : 0;
  return { data: out, shape: input.shape };
}

function add(a: Tensor, b: Tensor): Tensor {
  const out = new Float32Array(a.data.length);
  for (let i = 0; i < a.data.length; i++) out[i] = a.data[i] + b.data[i];
  return { data: out, shape: a.shape };
}

function linear(input: Float32Array, weight: Float32Array, bias: Float32Array, outFeatures: number, inFeatures: number): Float32Array {
  const out = new Float32Array(outFeatures);
  for (let o = 0; o < outFeatures; o++) {
    let sum = bias[o];
    const wRow = o * inFeatures;
    for (let i = 0; i < inFeatures; i++) sum += weight[wRow + i] * input[i];
    out[o] = sum;
  }
  return out;
}

function softmax(x: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  let max = -Infinity;
  for (let i = 0; i < x.length; i++) if (x[i] > max) max = x[i];
  let sum = 0;
  for (let i = 0; i < x.length; i++) {
    out[i] = Math.exp(x[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < x.length; i++) out[i] /= sum;
  return out;
}

function copyTensor(t: Tensor): Tensor {
  return { data: t.data.slice(), shape: t.shape.slice() };
}

// ── Model ─────────────────────────────────────────────────────────────────

export class Model {
  readonly manifest: Manifest;
  private weights: Weights;
  private numResBlocks: number;
  private numHidden: number;

  constructor(manifest: Manifest, buffer: ArrayBuffer) {
    this.manifest = manifest;
    this.weights = new Weights(manifest, buffer);
    this.numResBlocks = manifest.model_args.num_res_blocks;
    this.numHidden = manifest.model_args.num_hidden;
  }

  /**
   * Run one forward pass. `encodedState` is what Connect4.encodeState()
   * returns: a flat (3, 6, 7) Float32Array.
   *
   * If `captureActivations` is false (the default — used by MCTS for speed),
   * the returned `activations` map is empty. Set it to true when rendering
   * the network-internals visualization.
   */
  forward(encodedState: Float32Array, captureActivations = false): ForwardResult {
    const W = this.weights;
    const eps = this.manifest.bn_eps;
    const { row_count: rows, column_count: cols, num_channels: cin, action_size: actionSize } = this.manifest.game;
    const acts = new Map<string, Tensor>();
    const cap = (name: string, t: Tensor) => { if (captureActivations) acts.set(name, copyTensor(t)); };

    let x: Tensor = { data: encodedState, shape: [cin, rows, cols] };
    cap("input", x);

    // ── Input block: conv 3→hidden, BN, ReLU ───────────────────────────
    x = conv2d(x, W.get("input_block.0.weight"), W.get("input_block.0.bias"), this.numHidden, cin, 3, 3, 1);
    x = bn(x,
      W.get("input_block.1.weight"), W.get("input_block.1.bias"),
      W.get("input_block.1.running_mean"), W.get("input_block.1.running_var"), eps,
    );
    x = relu(x);
    cap("input_block_out", x);

    // ── Residual blocks ────────────────────────────────────────────────
    for (let i = 0; i < this.numResBlocks; i++) {
      const skip = x;
      const p = `res_blocks.${i}`;
      x = conv2d(x, W.get(`${p}.conv1.weight`), W.get(`${p}.conv1.bias`), this.numHidden, this.numHidden, 3, 3, 1);
      x = bn(x, W.get(`${p}.bn1.weight`), W.get(`${p}.bn1.bias`), W.get(`${p}.bn1.running_mean`), W.get(`${p}.bn1.running_var`), eps);
      x = relu(x);
      x = conv2d(x, W.get(`${p}.conv2.weight`), W.get(`${p}.conv2.bias`), this.numHidden, this.numHidden, 3, 3, 1);
      x = bn(x, W.get(`${p}.bn2.weight`), W.get(`${p}.bn2.bias`), W.get(`${p}.bn2.running_mean`), W.get(`${p}.bn2.running_var`), eps);
      x = add(x, skip);
      x = relu(x);
      cap(`res_block_${i}_out`, x);
    }

    // ── Policy head: 1×1 conv → BN → ReLU → flatten → linear ──────────
    const polCW = W.get("policy_head.0.weight");
    const polChannels = polCW.length / this.numHidden; // weight shape is [polC, hidden, 1, 1]
    let p: Tensor = conv2d(x, polCW, W.get("policy_head.0.bias"), polChannels, this.numHidden, 1, 1, 0);
    p = bn(p, W.get("policy_head.1.weight"), W.get("policy_head.1.bias"), W.get("policy_head.1.running_mean"), W.get("policy_head.1.running_var"), eps);
    p = relu(p);
    cap("policy_head_pre_linear", p);
    const policyLogits = linear(p.data, W.get("policy_head.4.weight"), W.get("policy_head.4.bias"), actionSize, polChannels * rows * cols);

    // ── Value head: 1×1 conv → BN → ReLU → flatten → linear → tanh ────
    const valCW = W.get("value_head.0.weight");
    const valChannels = valCW.length / this.numHidden;
    let v: Tensor = conv2d(x, valCW, W.get("value_head.0.bias"), valChannels, this.numHidden, 1, 1, 0);
    v = bn(v, W.get("value_head.1.weight"), W.get("value_head.1.bias"), W.get("value_head.1.running_mean"), W.get("value_head.1.running_var"), eps);
    v = relu(v);
    cap("value_head_pre_linear", v);
    const vScalar = linear(v.data, W.get("value_head.4.weight"), W.get("value_head.4.bias"), 1, valChannels * rows * cols);

    return {
      policyLogits,
      policy: softmax(policyLogits),
      value: Math.tanh(vScalar[0]),
      activations: acts,
    };
  }
}

export async function loadModel(manifestUrl: string, weightsUrl: string): Promise<Model> {
  const [manifestResp, blobResp] = await Promise.all([fetch(manifestUrl), fetch(weightsUrl)]);
  if (!manifestResp.ok) throw new Error(`Failed to fetch ${manifestUrl}: ${manifestResp.status}`);
  if (!blobResp.ok) throw new Error(`Failed to fetch ${weightsUrl}: ${blobResp.status}`);
  const manifest = (await manifestResp.json()) as Manifest;
  const buffer = await blobResp.arrayBuffer();
  return new Model(manifest, buffer);
}
