import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  render as rtlRender,
  cleanup,
  fireEvent,
} from "@testing-library/react";
import { I18nextProvider } from "react-i18next";
import type { ReactElement } from "react";
import type { InferenceBox, InferenceResult } from "@common/types";
import {
  useInferenceStore,
  boxKey,
  type CamBoxResult,
} from "@stores/useInferenceStore";
import i18n from "../../i18n";
import ExplainToggles from "../ExplainToggles";

const render = (ui: ReactElement) =>
  rtlRender(<I18nextProvider i18n={i18n}>{ui}</I18nextProvider>);

const RK = "0:model-a";

const makeBox = (boxId: string): InferenceBox => ({
  inferenceId: "inf-1",
  boxId,
  classId: "class-1",
  label: "wheat",
  isVerified: false,
  bboxSource: "model",
  topX: 0,
  topY: 0,
  bottomX: 10,
  bottomY: 10,
});

const makeResult = (boxIds: string[]): InferenceResult => ({
  scores: [],
  classifications: [],
  boxes: boxIds.map(makeBox),
  topN: [],
  overlapping: [],
  overlappingIndices: [],
  labelOccurrence: {},
  totalBoxes: boxIds.length,
  models: [],
  completedAt: "2024-01-01T00:00:00Z",
  isActive: true,
  minBoxSize: 10,
});

const makeCam = (classCount: number, grid = 2): CamBoxResult => ({
  grid,
  classes: Array.from({ length: classCount }, (_, i) => ({
    classIndex: i,
    label: `species-${i}`,
    score: 1 - i * 0.1,
    heatmap: Array.from({ length: grid * grid }, () => 0),
  })),
});

const seedCam = (boxId: string, classCount: number) => {
  const cams = new Map(useInferenceStore.getState().camResults);
  cams.set(boxKey(0, "model-a", boxId), makeCam(classCount));
  useInferenceStore.setState({ camResults: cams });
};

describe("ExplainToggles", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("en");
    useInferenceStore.setState({
      camResults: new Map(),
      camRank: new Map(),
      dffResults: new Map(),
      dffK: new Map(),
      dffActiveConcept: new Map(),
      explainMode: new Map(),
      dffPending: new Set(),
      requestDff: null,
      activeResultKey: null,
    });
  });

  afterEach(cleanup);

  it("renders nothing when the run has no explainability data", () => {
    const { container } = render(
      <ExplainToggles resultKey={RK} result={makeResult(["box-1"])} />,
    );
    expect(
      container.querySelector('[data-testid="explain-toggles"]'),
    ).toBeNull();
  });

  it("defaults to CAM mode and shows the rank toggles", () => {
    seedCam("box-1", 2);
    const { getByTestId, queryByTestId } = render(
      <ExplainToggles resultKey={RK} result={makeResult(["box-1"])} />,
    );
    expect(getByTestId("explain-toggles")).toBeInTheDocument();
    expect(getByTestId("cam-rank-0")).toBeInTheDocument();
    expect(queryByTestId("dff-k-value")).toBeNull();
  });

  it("switches to DFF mode and shows the concept controls", () => {
    seedCam("box-1", 2);
    const { getByTestId, queryByTestId } = render(
      <ExplainToggles resultKey={RK} result={makeResult(["box-1"])} />,
    );
    fireEvent.click(getByTestId("explain-mode-dff"));
    expect(useInferenceStore.getState().explainMode.get(RK)).toBe("dff");
    expect(getByTestId("dff-k-value")).toBeInTheDocument();
    expect(queryByTestId("cam-rank-0")).toBeNull();
  });
});
