// In-browser inference for the nanogpt checkpoints: the forward pass of model.py
// (GPT-2 blocks with pre-LayerNorm, exact GELU, tied embeddings, no biases unless
// the checkpoint has them) for one sequence at a time, with a key/value cache.
// export_web.py writes the weights this file reads. Runs in browsers and in Node
// (tests/test_web.py compares it with PyTorch).
(function (root) {
  "use strict";

  const SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<ans>"];
  const PAD = 0, BOS = 1, EOS = 2, ANS = 3;

  function decodeFloat32(b64) {
    const bytes = typeof atob === "function"
      ? Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))
      : new Uint8Array(Buffer.from(b64, "base64"));
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }

  class Tokenizer {
    constructor(alphabet) {
      this.alphabet = Array.from(alphabet);
      this.stoi = new Map(this.alphabet.map((ch, i) => [ch, i + SPECIAL_TOKENS.length]));
      this.vocabSize = SPECIAL_TOKENS.length + this.alphabet.length;
    }

    // Returns the ids, or throws with the offending characters listed.
    encode(text) {
      const ids = [], unknown = new Set();
      for (const ch of text) {
        const id = this.stoi.get(ch);
        if (id === undefined) unknown.add(ch); else ids.push(id);
      }
      if (unknown.size) {
        const err = new Error("outside the model's alphabet: " + [...unknown].map((c) => JSON.stringify(c)).join(" "));
        err.unknown = [...unknown];
        throw err;
      }
      return ids;
    }

    decode(ids) {
      let out = "";
      for (const id of ids) if (id >= SPECIAL_TOKENS.length) out += this.alphabet[id - SPECIAL_TOKENS.length];
      return out;
    }
  }

  // erf by Abramowitz & Stegun 7.1.26 (absolute error below 1.5e-7, far under
  // float32 rounding of the activations it feeds).
  function erf(x) {
    const s = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }

  function layerNorm(x, w, b, out) {
    const n = x.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let v = 0;
    for (let i = 0; i < n; i++) v += (x[i] - mean) * (x[i] - mean);
    const inv = 1 / Math.sqrt(v / n + 1e-5);
    for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + (b ? b[i] : 0);
    return out;
  }

  // out[r] = sum_c W[r, c] * x[c] (+ bias[r]) for a row-major (rows x cols) W.
  function matvec(W, x, rows, cols, out, bias) {
    for (let r = 0; r < rows; r++) {
      let acc = bias ? bias[r] : 0;
      const off = r * cols;
      for (let c = 0; c < cols; c++) acc += W[off + c] * x[c];
      out[r] = acc;
    }
    return out;
  }

  // Deterministic sampling (mulberry32), so a seed reproduces a sample.
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  class GPT {
    constructor(bundle) {
      const c = bundle.config;
      this.config = c;
      this.tokenizer = new Tokenizer(bundle.alphabet);
      const t = {};
      for (const [name, entry] of Object.entries(bundle.tensors)) t[name] = decodeFloat32(entry.data);
      const get = (name) => t[name] || null;
      this.wte = get("transformer.wte.weight");
      this.wpe = get("transformer.wpe.weight");
      this.lnf = [get("transformer.ln_f.weight"), get("transformer.ln_f.bias")];
      this.layers = [];
      for (let l = 0; l < c.n_layer; l++) {
        const p = "transformer.h." + l + ".";
        this.layers.push({
          ln1: [get(p + "ln_1.weight"), get(p + "ln_1.bias")],
          attn: [get(p + "attn.c_attn.weight"), get(p + "attn.c_attn.bias")],
          proj: [get(p + "attn.c_proj.weight"), get(p + "attn.c_proj.bias")],
          ln2: [get(p + "ln_2.weight"), get(p + "ln_2.bias")],
          fc: [get(p + "mlp.c_fc.weight"), get(p + "mlp.c_fc.bias")],
          fcProj: [get(p + "mlp.c_proj.weight"), get(p + "mlp.c_proj.bias")],
        });
      }
      const C = c.n_embd;
      this.buf = {
        x: new Float32Array(C), h: new Float32Array(C), qkv: new Float32Array(3 * C), y: new Float32Array(C),
        o: new Float32Array(C), fc: new Float32Array(4 * C), att: new Float32Array(c.block_size),
        logits: new Float32Array(c.vocab_size),
      };
    }

    newState() {
      const c = this.config;
      return {
        pos: 0,
        k: this.layers.map(() => new Float32Array(c.block_size * c.n_embd)),
        v: this.layers.map(() => new Float32Array(c.block_size * c.n_embd)),
      };
    }

    // Feeds one token and returns the logits for the next one (a view that the
    // next call overwrites).
    step(state, token) {
      const c = this.config, C = c.n_embd, H = c.n_head, D = C / H, T = state.pos;
      if (T >= c.block_size) throw new Error("context is full (" + c.block_size + " tokens)");
      const { x, h, qkv, y, o, fc, att, logits } = this.buf;
      for (let i = 0; i < C; i++) x[i] = this.wte[token * C + i] + this.wpe[T * C + i];
      const scale = 1 / Math.sqrt(D);
      this.layers.forEach((L, l) => {
        layerNorm(x, L.ln1[0], L.ln1[1], h);
        matvec(L.attn[0], h, 3 * C, C, qkv, L.attn[1]);
        const K = state.k[l], V = state.v[l];
        K.set(qkv.subarray(C, 2 * C), T * C);
        V.set(qkv.subarray(2 * C, 3 * C), T * C);
        for (let hd = 0; hd < H; hd++) {
          const off = hd * D;
          let max = -Infinity;
          for (let j = 0; j <= T; j++) {
            let s = 0;
            for (let d = 0; d < D; d++) s += qkv[off + d] * K[j * C + off + d];
            att[j] = s * scale;
            if (att[j] > max) max = att[j];
          }
          let sum = 0;
          for (let j = 0; j <= T; j++) { att[j] = Math.exp(att[j] - max); sum += att[j]; }
          for (let d = 0; d < D; d++) {
            let acc = 0;
            for (let j = 0; j <= T; j++) acc += att[j] * V[j * C + off + d];
            y[off + d] = acc / sum;
          }
        }
        matvec(L.proj[0], y, C, C, o, L.proj[1]);
        for (let i = 0; i < C; i++) x[i] += o[i];
        layerNorm(x, L.ln2[0], L.ln2[1], h);
        matvec(L.fc[0], h, 4 * C, C, fc, L.fc[1]);
        for (let i = 0; i < 4 * C; i++) fc[i] = 0.5 * fc[i] * (1 + erf(fc[i] / Math.SQRT2));
        matvec(L.fcProj[0], fc, C, 4 * C, o, L.fcProj[1]);
        for (let i = 0; i < C; i++) x[i] += o[i];
      });
      layerNorm(x, this.lnf[0], this.lnf[1], h);
      matvec(this.wte, h, c.vocab_size, C, logits, null);
      state.pos = T + 1;
      return logits;
    }

    static pick(logits, temperature, topK, random) {
      if (!(temperature > 0)) {
        let best = 0;
        for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
        return best;
      }
      const idx = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
      const keep = topK ? idx.slice(0, topK) : idx;
      const m = logits[keep[0]];
      const w = keep.map((i) => Math.exp((logits[i] - m) / temperature));
      let r = random() * w.reduce((a, b) => a + b, 0);
      for (let k = 0; k < keep.length; k++) { r -= w[k]; if (r <= 0) return keep[k]; }
      return keep[keep.length - 1];
    }

    // Completes a task prompt, <bos> prompt <ans> ... <eos>. `onToken(text)` is
    // called with the answer so far after every token and may return a promise
    // (to let the page repaint). Resolves to {answer, finished}, plus
    // malformed: true if the model emitted a special token mid-answer.
    async complete(prompt, { maxNewTokens = 128, temperature = 0, topK = null, seed = 0, onToken = null } = {}) {
      const ids = [BOS, ...this.tokenizer.encode(prompt), ANS];
      if (ids.length >= this.config.block_size) throw new Error("the prompt is longer than the model's context");
      const state = this.newState();
      let logits;
      for (const id of ids) logits = this.step(state, id);
      const random = rng(seed), out = [];
      const budget = Math.min(maxNewTokens, this.config.block_size - ids.length);
      for (let n = 0; n < budget; n++) {
        const next = GPT.pick(logits, temperature, topK, random);
        if (next === EOS) return { answer: this.tokenizer.decode(out), finished: true };
        // Any other special token inside an answer makes it malformed, as in
        // common.decode_response: the text shown must be the text produced.
        if (next < SPECIAL_TOKENS.length) return { answer: this.tokenizer.decode(out), finished: false, malformed: true };
        out.push(next);
        if (onToken) await onToken(this.tokenizer.decode(out));
        if (n + 1 < budget) logits = this.step(state, next);
      }
      return { answer: this.tokenizer.decode(out), finished: false };
    }
  }

  const api = { GPT, Tokenizer, SPECIAL_TOKENS, PAD, BOS, EOS, ANS };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.NanoGPT = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
