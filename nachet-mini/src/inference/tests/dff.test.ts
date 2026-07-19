import { describe, it, expect } from "vitest";
import { computeDffGroup } from "../dff";

// Two synthetic "parts" in a 6-channel feature space:
//   part A lights up channels {0,1,2}, part B lights up channels {3,4,5}.
// A token is either a part-A token or a part-B token. Because DFF factors the
// whole group jointly, concept indices must be shared across seeds — the concept
// that captures part A on seed 1 must be the same concept index on seed 2.
const CHANNELS = 6;
const TOKENS = 4; // 2×2 grid

const PART_A = { 0: 1.0, 1: 0.9, 2: 1.1 } as Record<number, number>;
const PART_B = { 3: 1.0, 4: 1.1, 5: 0.9 } as Record<number, number>;

/** Build one seed's (tokens × channels) features from a per-token part list. */
const makeSeed = (parts: Array<"A" | "B">): Float32Array => {
  const f = new Float32Array(TOKENS * CHANNELS);
  parts.forEach((part, t) => {
    const active = part === "A" ? PART_A : PART_B;
    for (const [ch, v] of Object.entries(active)) {
      f[t * CHANNELS + Number(ch)] = v;
    }
  });
  return f;
};

/** Per-token argmax concept index across the K heatmaps of one seed. */
const argmaxConcepts = (heatmaps: Float32Array[]): number[] => {
  const tokens = heatmaps[0].length;
  const out: number[] = [];
  for (let t = 0; t < tokens; t++) {
    let best = 0;
    let bestV = -Infinity;
    heatmaps.forEach((hm, k) => {
      if (hm[t] > bestV) {
        bestV = hm[t];
        best = k;
      }
    });
    out.push(best);
  }
  return out;
};

describe("computeDffGroup", () => {
  it("returns K heatmaps per seed on the right grid", () => {
    const seed = makeSeed(["A", "A", "B", "B"]);
    const res = computeDffGroup([seed], TOKENS, CHANNELS, { k: 2 });
    expect(res.k).toBe(2);
    expect(res.grid).toBe(2);
    expect(res.seeds).toHaveLength(1);
    expect(res.seeds[0].heatmaps).toHaveLength(2);
    expect(res.seeds[0].heatmaps[0]).toHaveLength(TOKENS);
    // Normalized to [0, 1].
    for (const hm of res.seeds[0].heatmaps)
      for (const v of hm) expect(v).toBeGreaterThanOrEqual(0);
  });

  it("shares concept indices across seeds (cross-seed consistency)", () => {
    // Same two parts, different spatial layout per seed.
    const seed1 = makeSeed(["A", "A", "B", "B"]);
    const seed2 = makeSeed(["A", "B", "A", "B"]);
    const res = computeDffGroup([seed1, seed2], TOKENS, CHANNELS, { k: 2 });

    const seg1 = argmaxConcepts(res.seeds[0].heatmaps);
    const seg2 = argmaxConcepts(res.seeds[1].heatmaps);

    // Within seed 1: the two A tokens agree, the two B tokens agree, and A ≠ B.
    expect(seg1[0]).toBe(seg1[1]);
    expect(seg1[2]).toBe(seg1[3]);
    expect(seg1[0]).not.toBe(seg1[2]);

    // The concept index that captured part A on seed 1 also captures part A on
    // seed 2 (tokens 0 and 2), and part B (tokens 1, 3) gets the other index.
    const conceptA = seg1[0];
    const conceptB = seg1[2];
    expect(seg2[0]).toBe(conceptA);
    expect(seg2[2]).toBe(conceptA);
    expect(seg2[1]).toBe(conceptB);
    expect(seg2[3]).toBe(conceptB);
  });

  it("is deterministic across runs", () => {
    const seeds = [
      makeSeed(["A", "A", "B", "B"]),
      makeSeed(["A", "B", "A", "B"]),
    ];
    const a = computeDffGroup(seeds, TOKENS, CHANNELS, { k: 2 });
    const b = computeDffGroup(seeds, TOKENS, CHANNELS, { k: 2 });
    for (let s = 0; s < a.seeds.length; s++)
      for (let k = 0; k < a.k; k++)
        expect(Array.from(a.seeds[s].heatmaps[k])).toEqual(
          Array.from(b.seeds[s].heatmaps[k]),
        );
  });

  it("clamps K to the rank of the group", () => {
    const seed = makeSeed(["A", "A", "B", "B"]);
    // Ask for more concepts than channels/positions allow.
    const res = computeDffGroup([seed], TOKENS, CHANNELS, { k: 10 });
    expect(res.k).toBeLessThanOrEqual(Math.min(CHANNELS, TOKENS));
    expect(res.seeds[0].heatmaps).toHaveLength(res.k);
  });

  it("rejects a non-square token grid", () => {
    const seed = new Float32Array(3 * CHANNELS);
    expect(() => computeDffGroup([seed], 3, CHANNELS, { k: 2 })).toThrow();
  });
});
