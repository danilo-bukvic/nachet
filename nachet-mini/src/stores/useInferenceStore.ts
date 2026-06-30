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

interface InferenceState {
  /** Results keyed by "imageIndex:modelConfigId" */
  results: Map<string, InferenceResult>;
  /** CAM maps keyed by "imageIndex:modelConfigId:boxId" */
  camResults: Map<string, CamBoxResult>;
  /**
   * Which class's CAM is currently overlaid, per box. Keyed by the box key →
   * class index. One species at a time per box (overlaying class maps doesn't
   * stack meaningfully); different boxes may show different species at once.
   */
  camVisible: Map<string, number>;
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
  /** Toggle a class's CAM overlay for a box (clears it if that class is on). */
  toggleCamClass: (key: string, classIndex: number) => void;
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
  camVisible: new Map(),
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

  toggleCamClass: (key: string, classIndex: number) => {
    set((state) => {
      const next = new Map(state.camVisible);
      if (next.get(key) === classIndex) next.delete(key);
      else next.set(key, classIndex);
      return { camVisible: next };
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
      const newCam = new Map(state.camResults);
      for (const key of newCam.keys()) {
        if (key.startsWith(prefix)) {
          newCam.delete(key);
        }
      }
      const newVisible = new Map(state.camVisible);
      for (const key of newVisible.keys()) {
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
        camResults: newCam,
        camVisible: newVisible,
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
      camResults: new Map(),
      camVisible: new Map(),
      activeResultKey: null,
      status: "idle",
      modelLoadProgress: null,
      error: null,
    });
  },
}));
