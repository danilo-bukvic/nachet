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

/** Per-box DFF key: "imageIndex:modelConfigId:boxId" */
export const dffKey = (
  imageIndex: number,
  modelConfigId: string,
  boxId: string,
): string => `${resultKey(imageIndex, modelConfigId)}:${boxId}`;

/** Deep Feature Factorization concept heatmaps for one classified box. */
export interface DffBoxResult {
  /** spatial grid side (e.g. 12 → 12×12). */
  grid: number;
  /** K concept heatmaps, each `grid*grid` floats in [0, 1]. */
  heatmaps: number[][];
}

interface InferenceState {
  /** Results keyed by "imageIndex:modelConfigId" */
  results: Map<string, InferenceResult>;
  /** DFF concept heatmaps keyed by "imageIndex:modelConfigId:boxId" */
  dffResults: Map<string, DffBoxResult>;
  /** dffKeys whose concept map is currently shown on the image (default: off) */
  dffVisible: Set<string>;
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
  setDffResult: (
    imageIndex: number,
    modelConfigId: string,
    boxId: string,
    dff: DffBoxResult,
  ) => void;
  getDffResult: (
    imageIndex: number,
    modelConfigId: string,
    boxId: string,
  ) => DffBoxResult | undefined;
  /** Toggle the concept-map overlay for one box (full dffKey). */
  toggleDffVisible: (key: string) => void;
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
  dffResults: new Map(),
  dffVisible: new Set(),
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

  setDffResult: (
    imageIndex: number,
    modelConfigId: string,
    boxId: string,
    dff: DffBoxResult,
  ) => {
    const key = dffKey(imageIndex, modelConfigId, boxId);
    set((state) => {
      const newMap = new Map(state.dffResults);
      newMap.set(key, dff);
      return { dffResults: newMap };
    });
  },

  getDffResult: (imageIndex: number, modelConfigId: string, boxId: string) => {
    return get().dffResults.get(dffKey(imageIndex, modelConfigId, boxId));
  },

  toggleDffVisible: (key: string) => {
    set((state) => {
      const next = new Set(state.dffVisible);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return { dffVisible: next };
    });
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
      const newDff = new Map(state.dffResults);
      for (const key of newDff.keys()) {
        if (key.startsWith(prefix)) {
          newDff.delete(key);
        }
      }
      const newVisible = new Set(state.dffVisible);
      for (const key of newVisible) {
        if (key.startsWith(prefix)) {
          newVisible.delete(key);
        }
      }
      const activeKey =
        state.activeResultKey?.startsWith(prefix) === true
          ? null
          : state.activeResultKey;
      return {
        results: newMap,
        dffResults: newDff,
        dffVisible: newVisible,
        activeResultKey: activeKey,
      };
    });
  },

  removeResult: (key: string) => {
    set((state) => {
      const newMap = new Map(state.results);
      newMap.delete(key);
      const activeKey =
        state.activeResultKey === key ? null : state.activeResultKey;
      return { results: newMap, activeResultKey: activeKey };
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
      dffResults: new Map(),
      dffVisible: new Set(),
      activeResultKey: null,
      status: "idle",
      modelLoadProgress: null,
      error: null,
    });
  },
}));
