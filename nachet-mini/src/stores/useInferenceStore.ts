import { create } from "zustand";
import type { InferenceResult } from "@common/types";

export type InferenceStatus =
  | "idle"
  | "loading-model"
  | "detecting"
  | "classifying"
  | "complete"
  | "error";

export interface ModelLoadProgress {
  name: string;
  progress: number;
}

/** Build the composite key used to store results: "imageIndex:modelConfigId" */
export const resultKey = (imageIndex: number, modelConfigId: string): string =>
  `${imageIndex}:${modelConfigId}`;

/** Per-box key: "imageIndex:modelConfigId:boxId" */
export const boxKey = (
  imageIndex: number,
  modelConfigId: string,
  boxId: string,
): string => `${resultKey(imageIndex, modelConfigId)}:${boxId}`;

interface DffMaps {
  dffResults: Map<string, DffRunResult>;
  dffK: Map<string, number>;
  dffActiveConcept: Map<string, number>;
  explainMode: Map<string, ExplainMode>;
  dffPending: Set<string>;
}

/** Copy the DFF maps, dropping every entry whose key starts with `prefix`. */
const pruneDffByPrefix = (state: DffMaps, prefix: string): DffMaps => {
  const dffResults = new Map(state.dffResults);
  const dffK = new Map(state.dffK);
  const dffActiveConcept = new Map(state.dffActiveConcept);
  const explainMode = new Map(state.explainMode);
  const dffPending = new Set(state.dffPending);
  for (const key of dffResults.keys())
    if (key.startsWith(prefix)) dffResults.delete(key);
  for (const key of dffK.keys()) if (key.startsWith(prefix)) dffK.delete(key);
  for (const key of dffActiveConcept.keys())
    if (key.startsWith(prefix)) dffActiveConcept.delete(key);
  for (const key of explainMode.keys())
    if (key.startsWith(prefix)) explainMode.delete(key);
  for (const key of dffPending)
    if (key.startsWith(prefix)) dffPending.delete(key);
  return { dffResults, dffK, dffActiveConcept, explainMode, dffPending };
};

/** One top-K class's CAM for a box. */
export interface CamClass {
  classIndex: number;
  label: string;
  score: number;
  /** grid*grid floats in [0, 1] (row-major). */
  heatmap: number[];
}

/** Class Activation Maps for one classified box. */
export interface CamBoxResult {
  /** spatial grid side (e.g. 12 → 12×12). */
  grid: number;
  /** one entry per top-K class. */
  classes: CamClass[];
}

/** Default / bounds for the DFF concept count (K). */
export const DEFAULT_DFF_K = 3;
export const MIN_DFF_K = 1;
export const MAX_DFF_K = 6;

/** DFF concept heatmaps for one box: K arrays of grid*grid floats in [0, 1]. */
export interface DffBoxResult {
  boxId: string;
  heatmaps: number[][];
}

/** One species group's DFF — the run's seeds that share a predicted species. */
export interface DffGroup {
  species: string;
  boxes: DffBoxResult[];
}

/** Deep Feature Factorization for one run (per species group, K shared concepts). */
export interface DffRunResult {
  /** Concept count actually used (may clamp below the request for tiny groups). */
  k: number;
  /** spatial grid side (e.g. 12 → 12×12). */
  grid: number;
  groups: DffGroup[];
}

/** Which explainability overlay a run shows. Absent in the map = "cam". */
export type ExplainMode = "cam" | "dff";

interface InferenceState {
  /** Results keyed by "imageIndex:modelConfigId" */
  results: Map<string, InferenceResult>;
  /** CAM maps keyed by "imageIndex:modelConfigId:boxId" */
  camResults: Map<string, CamBoxResult>;
  /**
   * Which prediction rank's CAM is overlaid, per run. Keyed by the result key
   * "imageIndex:modelConfigId" → rank index (0 = top-1, 1 = top-2, …). One rank
   * at a time (single-select); when set, every seed of that run shows its own
   * rank-N species map. Absent = no overlay.
   */
  camRank: Map<string, number>;
  /** DFF results keyed by the run key "imageIndex:modelConfigId". */
  dffResults: Map<string, DffRunResult>;
  /** Chosen concept count (K) per run; absent = DEFAULT_DFF_K. */
  dffK: Map<string, number>;
  /**
   * Highlighted concept per run for the DFF overlay. Absent = "all" (the full
   * per-token argmax segmentation); a number isolates that one concept.
   */
  dffActiveConcept: Map<string, number>;
  /** Explainability mode per run; absent = "cam". */
  explainMode: Map<string, ExplainMode>;
  /** Runs awaiting a `compute-dff` response (drives the DFF loading state). */
  dffPending: Set<string>;
  /**
   * Worker trigger for a DFF computation, registered by `useInference`. Kept in
   * the store so the toggle UI can request a (lazy) factorization without the
   * worker handle being prop-drilled through the view tree.
   */
  requestDff:
    | ((imageIndex: number, modelConfigId: string, k: number) => void)
    | null;
  /** Which result the user is currently viewing */
  activeResultKey: string | null;
  status: InferenceStatus;
  modelLoaded: boolean;
  modelLoadProgress: ModelLoadProgress | null;
  error: string | null;

  setResult: (
    imageIndex: number,
    modelConfigId: string,
    result: InferenceResult,
  ) => void;
  getResult: (
    imageIndex: number,
    modelConfigId: string,
  ) => InferenceResult | undefined;
  getResultsForImage: (
    imageIndex: number,
  ) => Array<{ modelConfigId: string; result: InferenceResult }>;
  setCamResult: (
    imageIndex: number,
    modelConfigId: string,
    boxId: string,
    cam: CamBoxResult,
  ) => void;
  /** Toggle a prediction rank's CAM overlay for a run (clears it if on). */
  toggleCamRank: (resultKey: string, rank: number) => void;
  setDffResult: (
    imageIndex: number,
    modelConfigId: string,
    dff: DffRunResult,
  ) => void;
  /** Set the concept count (K) for a run (clamped to [MIN_DFF_K, MAX_DFF_K]). */
  setDffK: (resultKey: string, k: number) => void;
  /** Highlight one concept (or `null` for the full segmentation). */
  setDffActiveConcept: (resultKey: string, concept: number | null) => void;
  /** Switch a run between the CAM and DFF overlays. */
  setExplainMode: (resultKey: string, mode: ExplainMode) => void;
  /** Register (or clear) the worker's DFF-compute trigger. */
  setRequestDff: (
    fn: ((imageIndex: number, modelConfigId: string, k: number) => void) | null,
  ) => void;
  /** Mark a run pending and ask the worker to factor it at `k` concepts. */
  triggerDff: (resultKey: string, k: number) => void;
  setActiveResultKey: (key: string | null) => void;
  removeResultsForImage: (imageIndex: number) => void;
  removeResult: (key: string) => void;
  setStatus: (status: InferenceStatus) => void;
  setModelLoaded: (value: boolean) => void;
  setModelLoadProgress: (progress: ModelLoadProgress | null) => void;
  setError: (error: string | null) => void;
  clearResults: () => void;
}

export const useInferenceStore = create<InferenceState>()((set, get) => ({
  results: new Map(),
  camResults: new Map(),
  camRank: new Map(),
  dffResults: new Map(),
  dffK: new Map(),
  dffActiveConcept: new Map(),
  explainMode: new Map(),
  dffPending: new Set(),
  requestDff: null,
  activeResultKey: null,
  status: "idle",
  modelLoaded: false,
  modelLoadProgress: null,
  error: null,

  setResult: (
    imageIndex: number,
    modelConfigId: string,
    result: InferenceResult,
  ) => {
    const key = resultKey(imageIndex, modelConfigId);
    set((state) => {
      const newMap = new Map(state.results);
      newMap.set(key, result);
      return { results: newMap };
    });
  },

  getResult: (imageIndex: number, modelConfigId: string) => {
    return get().results.get(resultKey(imageIndex, modelConfigId));
  },

  getResultsForImage: (imageIndex: number) => {
    const prefix = `${imageIndex}:`;
    const entries: Array<{ modelConfigId: string; result: InferenceResult }> =
      [];
    for (const [key, result] of get().results) {
      if (key.startsWith(prefix)) {
        entries.push({ modelConfigId: key.slice(prefix.length), result });
      }
    }
    return entries;
  },

  setCamResult: (
    imageIndex: number,
    modelConfigId: string,
    boxId: string,
    cam: CamBoxResult,
  ) => {
    const key = boxKey(imageIndex, modelConfigId, boxId);
    set((state) => {
      const newMap = new Map(state.camResults);
      newMap.set(key, cam);
      return { camResults: newMap };
    });
  },

  toggleCamRank: (key: string, rank: number) => {
    set((state) => {
      const next = new Map(state.camRank);
      if (next.get(key) === rank) next.delete(key);
      else next.set(key, rank);
      return { camRank: next };
    });
  },

  setDffResult: (
    imageIndex: number,
    modelConfigId: string,
    dff: DffRunResult,
  ) => {
    const key = resultKey(imageIndex, modelConfigId);
    set((state) => {
      const next = new Map(state.dffResults);
      next.set(key, dff);
      const pending = new Set(state.dffPending);
      pending.delete(key);
      return { dffResults: next, dffPending: pending };
    });
  },

  setDffK: (key: string, k: number) => {
    const clamped = Math.max(MIN_DFF_K, Math.min(MAX_DFF_K, Math.round(k)));
    set((state) => {
      const next = new Map(state.dffK);
      next.set(key, clamped);
      return { dffK: next };
    });
  },

  setDffActiveConcept: (key: string, concept: number | null) => {
    set((state) => {
      const next = new Map(state.dffActiveConcept);
      if (concept === null || next.get(key) === concept) next.delete(key);
      else next.set(key, concept);
      return { dffActiveConcept: next };
    });
  },

  setExplainMode: (key: string, mode: ExplainMode) => {
    set((state) => {
      const next = new Map(state.explainMode);
      next.set(key, mode);
      return { explainMode: next };
    });
  },

  setRequestDff: (fn) => {
    set({ requestDff: fn });
  },

  triggerDff: (key: string, k: number) => {
    const trigger = get().requestDff;
    if (!trigger) return;
    const sep = key.indexOf(":");
    if (sep < 0) return;
    const imageIndex = Number(key.slice(0, sep));
    const modelConfigId = key.slice(sep + 1);
    if (!Number.isFinite(imageIndex) || modelConfigId === "") return;
    set((state) => {
      const pending = new Set(state.dffPending);
      pending.add(key);
      return { dffPending: pending };
    });
    trigger(imageIndex, modelConfigId, k);
  },

  setActiveResultKey: (key: string | null) => {
    set({ activeResultKey: key });
  },

  removeResultsForImage: (imageIndex: number) => {
    const prefix = `${imageIndex}:`;
    set((state) => {
      const newMap = new Map(state.results);
      for (const key of newMap.keys()) {
        if (key.startsWith(prefix)) {
          newMap.delete(key);
        }
      }
      const newCam = new Map(state.camResults);
      for (const key of newCam.keys()) {
        if (key.startsWith(prefix)) {
          newCam.delete(key);
        }
      }
      const newRank = new Map(state.camRank);
      for (const key of newRank.keys()) {
        if (key.startsWith(prefix)) {
          newRank.delete(key);
        }
      }
      const dff = pruneDffByPrefix(state, prefix);
      const activeKey =
        state.activeResultKey?.startsWith(prefix) === true
          ? null
          : state.activeResultKey;
      return {
        results: newMap,
        camResults: newCam,
        camRank: newRank,
        ...dff,
        activeResultKey: activeKey,
      };
    });
  },

  removeResult: (key: string) => {
    set((state) => {
      const newMap = new Map(state.results);
      newMap.delete(key);
      // Also drop this run's CAM state, keyed by the "<resultKey>:<boxId>"
      // prefix (maps) and the resultKey itself (overlaid rank). Box ids are
      // reused across runs, so leaving these behind could resurface a stale
      // heatmap or overlay. Mirrors removeResultsForImage.
      const prefix = `${key}:`;
      const newCam = new Map(state.camResults);
      for (const camKey of newCam.keys()) {
        if (camKey.startsWith(prefix)) newCam.delete(camKey);
      }
      const newRank = new Map(state.camRank);
      newRank.delete(key);
      // DFF state is keyed by the run key itself (not "<key>:<boxId>"), so drop
      // exactly this run's entries — a prefix match could catch a sibling run
      // whose key shares this one as a leading substring.
      const dffResults = new Map(state.dffResults);
      dffResults.delete(key);
      const dffK = new Map(state.dffK);
      dffK.delete(key);
      const dffActiveConcept = new Map(state.dffActiveConcept);
      dffActiveConcept.delete(key);
      const explainMode = new Map(state.explainMode);
      explainMode.delete(key);
      const dffPending = new Set(state.dffPending);
      dffPending.delete(key);
      const activeKey =
        state.activeResultKey === key ? null : state.activeResultKey;
      return {
        results: newMap,
        camResults: newCam,
        camRank: newRank,
        dffResults,
        dffK,
        dffActiveConcept,
        explainMode,
        dffPending,
        activeResultKey: activeKey,
      };
    });
  },

  setStatus: (status: InferenceStatus) => {
    set({ status });
  },

  setModelLoaded: (value: boolean) => {
    set({ modelLoaded: value });
  },

  setModelLoadProgress: (progress: ModelLoadProgress | null) => {
    set({ modelLoadProgress: progress });
  },

  setError: (error: string | null) => {
    set({ error });
  },

  clearResults: () => {
    set({
      results: new Map(),
      camResults: new Map(),
      camRank: new Map(),
      dffResults: new Map(),
      dffK: new Map(),
      dffActiveConcept: new Map(),
      explainMode: new Map(),
      dffPending: new Set(),
      activeResultKey: null,
      status: "idle",
      modelLoadProgress: null,
      error: null,
    });
  },
}));
