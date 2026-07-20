/**
 * Deep Feature Factorization (DFF) in the browser — batch, per-species.
 *
 * DFF (Collins, Achanta, Süsstrunk, ECCV 2018) runs Non-negative Matrix
 * Factorization over the activations of a *set* of images to discover K
 * concepts (recurring parts) shared across that set. For the Swin classifier we
 * factor the `swin_layernorm` token features (channels = 1536, tokens = 144 per
 * seed) of every detected seed that shares a predicted species, so a concept
 * means the same sub-part on every seed of that species — e.g. K=1 separates
 * foreground from background, K>1 pulls apart body vs. appendage.
 *
 * This is the joint-batch generalization of the earlier per-seed DFF: instead
 * of one NMF per seed (whose concept indices had no cross-seed meaning), we
 * stack all seeds of a group into one matrix and factor once.
 *
 *   X (channels × tokens·N)  ≈  W (channels × K)  ·  H (K × tokens·N)
 *
 * W holds the K concept directions in channel space; each row of H is one
 * concept's activation across every token of every seed. Splitting H back per
 * seed gives, for each seed, K spatial heatmaps on the 12×12 token grid.
 *
 * The factorization is deterministic (fixed-seed randomized SVD for the NNDSVD
 * initialization + multiplicative updates), so concept indices — and therefore
 * their assigned colors — are stable across recomputations.
 */

export interface DffSeedMaps {
  /** Concept heatmaps for one seed: K arrays of `grid*grid` floats in [0, 1]. */
  heatmaps: Float32Array[];
}

export interface DffGroupResult {
  /** Number of concepts (NMF components). */
  k: number;
  /** Spatial grid side (12 for a 144-token Swin stage). */
  grid: number;
  /** One entry per input seed, in the same order the seeds were passed in. */
  seeds: DffSeedMaps[];
}

export interface DffOptions {
  /** Number of concepts (default 3). */
  k?: number;
  /** Multiplicative-update iterations (default 200). */
  iters?: number;
}

const DEFAULT_K = 3;
const DEFAULT_ITERS = 200;
const EPS = 1e-9;

/**
 * Factor a group of same-species seeds into shared concepts.
 *
 * @param seeds    one flat `swin_layernorm` tensor per seed, each laid out
 *                 (tokens, channels) row-major (i.e. `features[t * channels + ch]`).
 * @param tokens   spatial token count per seed (e.g. 144).
 * @param channels feature dimension (e.g. 1536); must be identical for all seeds.
 * @returns per-seed concept heatmaps, normalized on one shared scale across all
 *          concepts and seeds so a per-token argmax over concepts is meaningful.
 */
export function computeDffGroup(
  seeds: Float32Array[],
  tokens: number,
  channels: number,
  options: DffOptions = {},
): DffGroupResult {
  if (seeds.length === 0) throw new Error("DFF: no seeds in group");
  const grid = Math.round(Math.sqrt(tokens));
  if (grid * grid !== tokens) {
    throw new Error(`DFF: non-square token grid (${tokens})`);
  }
  const iters = options.iters ?? DEFAULT_ITERS;
  const C = channels;
  const N = seeds.length;
  const P = tokens * N; // total spatial positions across the group
  // K can't exceed the rank of X; clamp so the SVD/NMF stay well-defined for
  // tiny groups (e.g. a single seed with a low token count).
  const K = Math.max(1, Math.min(options.k ?? DEFAULT_K, C, P));

  // --- build X (C × P), clamping negatives to zero ---
  // Column (s, t) -> s * tokens + t. X[ch, col] = seeds[s][t * C + ch].
  //
  // NMF needs a non-negative matrix, and layernorm output is mixed-sign. The
  // obvious fix — shifting each channel up by its minimum — is a trap: it adds
  // a large constant to every entry, so the factorization spends its first (and
  // largest) component modelling that DC offset. That component is strongly
  // active at *every* location, wins the per-token argmax everywhere, and
  // leaves the real parts with no territory. Clamping instead keeps X sparse,
  // which is the regime where NMF actually yields a parts-based decomposition.
  const X = new Float64Array(C * P);
  for (let ch = 0; ch < C; ch++) {
    const xrow = ch * P;
    for (let s = 0; s < N; s++) {
      const seed = seeds[s];
      const base = s * tokens;
      for (let t = 0; t < tokens; t++) {
        const v = seed[t * C + ch];
        X[xrow + base + t] = v > 0 ? v : 0;
      }
    }
  }

  const { W, H } = nmf(X, C, P, K, iters);
  void W; // concept directions aren't needed downstream (no concept labeling)

  // --- normalize each concept by its own peak, then split per seed ---
  // Per concept rather than one shared maximum: concepts differ in intrinsic
  // magnitude, and under a single global scale the largest one can sit above
  // every other concept's ceiling and win the argmax at every location (the
  // rest then render blank). Scaling each concept to its own peak puts them on
  // equal footing, so the segmentation shows which part is most characteristic
  // of a location. The scale is shared across the group within a concept, so a
  // concept that is weaker on one seed than another still reads that way.
  const conceptScale = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    let mx = 0;
    const hrow = k * P;
    for (let c = 0; c < P; c++) if (H[hrow + c] > mx) mx = H[hrow + c];
    conceptScale[k] = mx || 1;
  }

  const seedResults: DffSeedMaps[] = [];
  for (let s = 0; s < N; s++) {
    const base = s * tokens;
    const heatmaps: Float32Array[] = [];
    for (let k = 0; k < K; k++) {
      const hm = new Float32Array(tokens);
      const hrow = k * P + base;
      const scale = conceptScale[k];
      for (let t = 0; t < tokens; t++) hm[t] = H[hrow + t] / scale;
      heatmaps.push(hm);
    }
    seedResults.push({ heatmaps });
  }

  return { k: K, grid, seeds: seedResults };
}

// ---------------------------------------------------------------------------
// NMF: NNDSVD init + multiplicative updates (deterministic)
// ---------------------------------------------------------------------------

/**
 * Factor X (N×M, non-negative) into W (N×K) · H (K×M) with multiplicative
 * updates from an NNDSVD initialization. Mirrors sklearn `NMF(solver='mu')`.
 */
function nmf(X: Float64Array, N: number, M: number, K: number, iters: number) {
  const { W, H } = nndsvdInit(X, N, M, K);
  for (let it = 0; it < iters; it++) {
    // H *= (WᵀX) / (WᵀW·H)
    const WtW = new Float64Array(K * K);
    for (let a = 0; a < K; a++)
      for (let b = a; b < K; b++) {
        let s = 0;
        for (let r = 0; r < N; r++) s += W[r * K + a] * W[r * K + b];
        WtW[a * K + b] = WtW[b * K + a] = s;
      }
    const WtX = new Float64Array(K * M);
    for (let r = 0; r < N; r++)
      for (let a = 0; a < K; a++) {
        const w = W[r * K + a];
        if (!w) continue;
        const xo = r * M;
        const ho = a * M;
        for (let c = 0; c < M; c++) WtX[ho + c] += w * X[xo + c];
      }
    for (let a = 0; a < K; a++)
      for (let c = 0; c < M; c++) {
        let d = 0;
        for (let b = 0; b < K; b++) d += WtW[a * K + b] * H[b * M + c];
        H[a * M + c] *= WtX[a * M + c] / (d + EPS);
      }
    // W *= (XHᵀ) / (W·HHᵀ)
    const HHt = new Float64Array(K * K);
    for (let a = 0; a < K; a++)
      for (let b = a; b < K; b++) {
        let s = 0;
        for (let c = 0; c < M; c++) s += H[a * M + c] * H[b * M + c];
        HHt[a * K + b] = HHt[b * K + a] = s;
      }
    const XHt = new Float64Array(N * K);
    for (let r = 0; r < N; r++) {
      const xo = r * M;
      for (let a = 0; a < K; a++) {
        let s = 0;
        const ho = a * M;
        for (let c = 0; c < M; c++) s += X[xo + c] * H[ho + c];
        XHt[r * K + a] = s;
      }
    }
    for (let r = 0; r < N; r++)
      for (let a = 0; a < K; a++) {
        let d = 0;
        for (let b = 0; b < K; b++) d += W[r * K + b] * HHt[b * K + a];
        W[r * K + a] *= XHt[r * K + a] / (d + EPS);
      }
  }
  return { W, H };
}

/**
 * NNDSVD initialization (Boutsidis & Gallopoulos) from the top-K singular
 * triplets of X. The old per-seed path got those triplets from a full Jacobi
 * eigendecomposition of the M×M Gram matrix; for a batch M = tokens·N grows, so
 * we use a deterministic randomized SVD (below) whose cost scales with K, not M².
 */
function nndsvdInit(X: Float64Array, N: number, M: number, K: number) {
  const { U, S, V } = randomizedSvd(X, N, M, K);
  const W = new Float64Array(N * K);
  const H = new Float64Array(K * M);

  for (let k = 0; k < K; k++) {
    const s = Math.sqrt(Math.max(S[k], 0));
    const u = (r: number) => U[r * K + k]; // left singular vector (length N)
    const v = (c: number) => V[c * K + k]; // right singular vector (length M)

    if (k === 0) {
      // Leading component: |u| and |v| (guaranteed dominant sign by Perron).
      const su = Math.sqrt(s);
      for (let r = 0; r < N; r++) W[r * K] = su * Math.abs(u(r));
      for (let c = 0; c < M; c++) H[c] = su * Math.abs(v(c));
      continue;
    }

    // Split each singular vector into its positive and negative parts and keep
    // whichever pairing carries more energy.
    const uCol = new Float64Array(N);
    const vCol = new Float64Array(M);
    for (let r = 0; r < N; r++) uCol[r] = u(r);
    for (let c = 0; c < M; c++) vCol[c] = v(c);
    const up = posPart(uCol, +1);
    const un = posPart(uCol, -1);
    const vp = posPart(vCol, +1);
    const vn = posPart(vCol, -1);
    const usePos = up.norm * vp.norm >= un.norm * vn.norm;
    const uu = usePos ? up : un;
    const vv = usePos ? vp : vn;
    const sigma = usePos ? up.norm * vp.norm : un.norm * vn.norm;
    const sc = Math.sqrt(s * sigma);
    for (let r = 0; r < N; r++)
      W[r * K + k] = uu.norm > 0 ? (sc * uu.vec[r]) / uu.norm : 0;
    for (let c = 0; c < M; c++)
      H[k * M + c] = vv.norm > 0 ? (sc * vv.vec[c]) / vv.norm : 0;
  }
  return { W, H };
}

/** Positive (`sign=+1`) or negated-positive (`sign=-1`) part of a vector. */
function posPart(v: Float64Array, sign: number) {
  const vec = new Float64Array(v.length);
  let s = 0;
  for (let i = 0; i < v.length; i++) {
    const x = sign > 0 ? Math.max(v[i], 0) : Math.max(-v[i], 0);
    vec[i] = x;
    s += x * x;
  }
  return { vec, norm: Math.sqrt(s) };
}

// ---------------------------------------------------------------------------
// Deterministic randomized top-K SVD of X (N×M, row-major)
// ---------------------------------------------------------------------------

/**
 * Approximate the top-K singular triplets of X via a randomized range finder
 * with a few power iterations (Halko, Martinsson & Tropp). Returns:
 *   U (N×K, orthonormal columns), S (length K, descending), V (M×K).
 *
 * A fixed-seed PRNG makes the sketch — and thus the whole factorization —
 * reproducible. Cost is O(N·M·Kp·iters), independent of forming any N² or M²
 * Gram matrix.
 */
function randomizedSvd(X: Float64Array, N: number, M: number, K: number) {
  const p = 4; // oversampling for a more accurate range estimate
  const Kp = Math.min(K + p, N, M);
  const rand = mulberry32(0x9e3779b9);

  // Sketch Y = X · Ω, Ω is M×Kp Gaussian → Y is N×Kp.
  const Omega = new Float64Array(M * Kp);
  for (let i = 0; i < Omega.length; i++) Omega[i] = gaussian(rand);
  let Q = matTimesSkinny(X, N, M, Omega, Kp); // N×Kp
  orthonormalize(Q, N, Kp);

  // Power iterations: Q ← orth(X · (Xᵀ · Q)) sharpens Q toward the top subspace.
  for (let it = 0; it < 4; it++) {
    const Z = matTTimesSkinny(X, N, M, Q, Kp); // M×Kp
    orthonormalize(Z, M, Kp);
    Q = matTimesSkinny(X, N, M, Z, Kp); // N×Kp
    orthonormalize(Q, N, Kp);
  }

  // Project: B = Qᵀ · X  (Kp×M). Then svd(B) via eigendecomposition of B·Bᵀ.
  const B = new Float64Array(Kp * M);
  for (let j = 0; j < Kp; j++)
    for (let c = 0; c < M; c++) {
      let s = 0;
      for (let r = 0; r < N; r++) s += Q[r * Kp + j] * X[r * M + c];
      B[j * M + c] = s;
    }
  const BBt = new Float64Array(Kp * Kp);
  for (let a = 0; a < Kp; a++)
    for (let b = a; b < Kp; b++) {
      let s = 0;
      for (let c = 0; c < M; c++) s += B[a * M + c] * B[b * M + c];
      BBt[a * Kp + b] = BBt[b * Kp + a] = s;
    }
  const { val, vec } = jacobi(BBt, Kp); // eigenvalues (=σ²) + eigenvectors (Ũ)
  const order = [...val.keys()].sort((i, j) => val[j] - val[i]).slice(0, K);

  const U = new Float64Array(N * K);
  const S = new Float64Array(K);
  const V = new Float64Array(M * K);
  for (let k = 0; k < K; k++) {
    const idx = order[k];
    const sigma = Math.sqrt(Math.max(val[idx], 0));
    S[k] = sigma;
    // U[:,k] = Q · Ũ[:,idx]
    for (let r = 0; r < N; r++) {
      let s = 0;
      for (let j = 0; j < Kp; j++) s += Q[r * Kp + j] * vec[j * Kp + idx];
      U[r * K + k] = s;
    }
    // V[:,k] = Bᵀ · Ũ[:,idx] / σ
    for (let c = 0; c < M; c++) {
      let s = 0;
      for (let j = 0; j < Kp; j++) s += B[j * M + c] * vec[j * Kp + idx];
      V[c * K + k] = sigma > 0 ? s / sigma : 0;
    }
  }
  return { U, S, V };
}

/** Y (N×Kp) = X (N×M) · A (M×Kp), all row-major. */
function matTimesSkinny(
  X: Float64Array,
  N: number,
  M: number,
  A: Float64Array,
  Kp: number,
): Float64Array {
  const Y = new Float64Array(N * Kp);
  for (let r = 0; r < N; r++) {
    const xo = r * M;
    const yo = r * Kp;
    for (let c = 0; c < M; c++) {
      const x = X[xo + c];
      if (!x) continue;
      const ao = c * Kp;
      for (let j = 0; j < Kp; j++) Y[yo + j] += x * A[ao + j];
    }
  }
  return Y;
}

/** Z (M×Kp) = Xᵀ (M×N) · A (N×Kp), all row-major. */
function matTTimesSkinny(
  X: Float64Array,
  N: number,
  M: number,
  A: Float64Array,
  Kp: number,
): Float64Array {
  const Z = new Float64Array(M * Kp);
  for (let r = 0; r < N; r++) {
    const xo = r * M;
    const ao = r * Kp;
    for (let c = 0; c < M; c++) {
      const x = X[xo + c];
      if (!x) continue;
      const zo = c * Kp;
      for (let j = 0; j < Kp; j++) Z[zo + j] += x * A[ao + j];
    }
  }
  return Z;
}

/** Modified Gram-Schmidt orthonormalization of the columns of Q (rows×cols). */
function orthonormalize(Q: Float64Array, rows: number, cols: number): void {
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < j; i++) {
      let dot = 0;
      for (let r = 0; r < rows; r++) dot += Q[r * cols + j] * Q[r * cols + i];
      for (let r = 0; r < rows; r++) Q[r * cols + j] -= dot * Q[r * cols + i];
    }
    let norm = 0;
    for (let r = 0; r < rows; r++) {
      const v = Q[r * cols + j];
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    if (norm > 1e-12) for (let r = 0; r < rows; r++) Q[r * cols + j] /= norm;
  }
}

/**
 * Cyclic Jacobi eigendecomposition of a symmetric (n×n) matrix (n small — the
 * randomized-SVD projection size Kp). Returns eigenvalues `val` and
 * eigenvectors `vec` (column k is the k-th vector, row-major).
 */
function jacobi(A: Float64Array, n: number) {
  const a = Float64Array.from(A);
  const vec = new Float64Array(n * n);
  for (let i = 0; i < n; i++) vec[i * n + i] = 1;
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
    if (off < 1e-20) break;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const phi = 0.5 * Math.atan2(2 * apq, a[q * n + q] - a[p * n + p]);
        const c = Math.cos(phi);
        const s = Math.sin(phi);
        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p];
          const akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k];
          const aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = vec[k * n + p];
          const vkq = vec[k * n + q];
          vec[k * n + p] = c * vkp - s * vkq;
          vec[k * n + q] = s * vkp + c * vkq;
        }
      }
  }
  const val = new Float64Array(n);
  for (let i = 0; i < n; i++) val[i] = a[i * n + i];
  return { val, vec };
}

/** Deterministic 32-bit PRNG (mulberry32) → floats in [0, 1). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard-normal sample via Box-Muller from a uniform generator. */
function gaussian(rand: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
