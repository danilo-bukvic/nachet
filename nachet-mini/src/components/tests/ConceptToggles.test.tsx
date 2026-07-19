import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  render as rtlRender,
  cleanup,
  fireEvent,
} from "@testing-library/react";
import { I18nextProvider } from "react-i18next";
import type { ReactElement } from "react";
import {
  useInferenceStore,
  type DffRunResult,
} from "@stores/useInferenceStore";
import i18n from "../../i18n";
import ConceptToggles from "../ConceptToggles";

const render = (ui: ReactElement) =>
  rtlRender(<I18nextProvider i18n={i18n}>{ui}</I18nextProvider>);

const RK = "0:model-a";

const makeDff = (k = 3, seeds = 2, grid = 2): DffRunResult => ({
  k,
  grid,
  groups: [
    {
      species: "wheat",
      boxes: Array.from({ length: seeds }, (_, i) => ({
        boxId: `box-${i}`,
        heatmaps: Array.from({ length: k }, () =>
          Array.from({ length: grid * grid }, () => 0),
        ),
      })),
    },
  ],
});

const resetStore = () =>
  useInferenceStore.setState({
    dffResults: new Map(),
    dffK: new Map(),
    dffActiveConcept: new Map(),
    explainMode: new Map(),
    dffPending: new Set(),
    requestDff: null,
    activeResultKey: null,
  });

describe("ConceptToggles", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("en");
    resetStore();
  });

  afterEach(cleanup);

  it("requests a factorization at the default K on mount and shows the loading state", () => {
    const requestDff = vi.fn();
    useInferenceStore.setState({ requestDff });
    const { getByTestId } = render(<ConceptToggles resultKey={RK} />);
    expect(requestDff).toHaveBeenCalledWith(0, "model-a", 3);
    expect(getByTestId("dff-computing")).toBeInTheDocument();
  });

  it("renders All parts, one row per concept, and the batch summary once results arrive", () => {
    useInferenceStore.setState({ dffResults: new Map([[RK, makeDff(3)]]) });
    const { getByTestId } = render(<ConceptToggles resultKey={RK} />);
    expect(getByTestId("dff-concept-all")).toBeInTheDocument();
    expect(getByTestId("dff-concept-0")).toHaveTextContent("Concept 1");
    expect(getByTestId("dff-concept-1")).toHaveTextContent("Concept 2");
    expect(getByTestId("dff-concept-2")).toHaveTextContent("Concept 3");
    const group = getByTestId("dff-group");
    expect(group).toHaveTextContent("wheat");
    expect(group).toHaveTextContent("2 seeds");
  });

  it("isolates a concept on click and returns to segmentation via All parts", () => {
    useInferenceStore.setState({ dffResults: new Map([[RK, makeDff(3)]]) });
    const { getByTestId } = render(<ConceptToggles resultKey={RK} />);
    fireEvent.click(getByTestId("dff-concept-1"));
    expect(useInferenceStore.getState().dffActiveConcept.get(RK)).toBe(1);
    fireEvent.click(getByTestId("dff-concept-all"));
    expect(useInferenceStore.getState().dffActiveConcept.has(RK)).toBe(false);
  });

  it("changes K via the stepper", () => {
    useInferenceStore.setState({
      dffResults: new Map([[RK, makeDff(3)]]),
      requestDff: vi.fn(),
    });
    const { getByLabelText, getByTestId } = render(
      <ConceptToggles resultKey={RK} />,
    );
    expect(getByTestId("dff-k-value")).toHaveTextContent("3");
    fireEvent.click(getByLabelText("more concepts"));
    expect(useInferenceStore.getState().dffK.get(RK)).toBe(4);
  });

  it("shows an empty note when the factorization returns no groups", () => {
    useInferenceStore.setState({
      dffResults: new Map([[RK, { k: 3, grid: 0, groups: [] }]]),
    });
    const { getByTestId, queryByTestId } = render(
      <ConceptToggles resultKey={RK} />,
    );
    expect(getByTestId("dff-empty")).toBeInTheDocument();
    expect(queryByTestId("dff-concept-all")).toBeNull();
  });
});
