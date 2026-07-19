import { Box, IconButton, Tooltip, CircularProgress } from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import RemoveIcon from "@mui/icons-material/Remove";
import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import {
  useInferenceStore,
  DEFAULT_DFF_K,
  MIN_DFF_K,
  MAX_DFF_K,
} from "@stores/useInferenceStore";
import { conceptColorCss } from "@common/dffColors";

/**
 * DFF concept controls shown under a run in the Images panel (DFF mode).
 *
 * A K stepper picks how many shared concepts to discover; "All parts" colors
 * each seed by its dominant concept (the segmentation view), and each concept
 * row isolates that one part across every seed of the species. The factorization
 * is requested lazily — mounting this (i.e. entering DFF mode) or changing K
 * asks the worker to factor the run's retained features.
 */

interface Props {
  /** "imageIndex:modelConfigId" for this run. */
  resultKey: string;
}

const rowSx = (on: boolean) => ({
  display: "flex",
  alignItems: "center",
  gap: "0.4vw",
  pl: "5.2vh",
  pr: "0.8vh",
  py: "0.35vh",
  fontSize: "1.25vh",
  cursor: "pointer",
  color: on ? "text.primary" : "text.secondary",
  backgroundColor: on ? "#E3F2FD" : "transparent",
  borderTop: "1px solid #f5f5f5",
  "&:hover": { backgroundColor: on ? "#E3F2FD" : "#F5F5F5" },
});

const ConceptToggles = ({ resultKey }: Props) => {
  const { t } = useTranslation("main");
  const dffResults = useInferenceStore((s) => s.dffResults);
  const dffK = useInferenceStore((s) => s.dffK);
  const dffActiveConcept = useInferenceStore((s) => s.dffActiveConcept);
  const dffPending = useInferenceStore((s) => s.dffPending);
  const setDffK = useInferenceStore((s) => s.setDffK);
  const setDffActiveConcept = useInferenceStore((s) => s.setDffActiveConcept);
  const triggerDff = useInferenceStore((s) => s.triggerDff);
  const setActiveResultKey = useInferenceStore((s) => s.setActiveResultKey);

  const currentK = dffK.get(resultKey) ?? DEFAULT_DFF_K;
  const dff = dffResults.get(resultKey);
  const pending = dffPending.has(resultKey);

  // Ask the worker to factor this run whenever we lack a result at the chosen K.
  // Setting `pending` (in triggerDff) makes the guard skip re-requests, and the
  // response either fills `dff` at `currentK` or clears pending — so this fires
  // exactly once per (run, K).
  useEffect(() => {
    if (pending) return;
    if (!dff || dff.k !== currentK) triggerDff(resultKey, currentK);
  }, [resultKey, currentK, dff, pending, triggerDff]);

  const conceptCount = dff?.groups[0]?.boxes[0]?.heatmaps.length ?? 0;
  const activeConcept = dffActiveConcept.get(resultKey);

  const changeK = (delta: number) => {
    const next = currentK + delta;
    if (next < MIN_DFF_K || next > MAX_DFF_K) return;
    setActiveResultKey(resultKey);
    setDffK(resultKey, next);
  };

  return (
    <Box>
      {/* Concept-count (K) stepper */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: "0.4vw",
          pl: "5.2vh",
          pr: "0.8vh",
          py: "0.3vh",
          fontSize: "1.25vh",
          color: "text.secondary",
          borderTop: "1px solid #f5f5f5",
        }}
      >
        <Box sx={{ flex: 1 }}>{t("dff.concepts")}</Box>
        <IconButton
          size="small"
          sx={{ padding: "0.2vh" }}
          aria-label={t("dff.decrease")}
          disabled={currentK <= MIN_DFF_K || pending}
          onClick={() => changeK(-1)}
        >
          <RemoveIcon sx={{ fontSize: "1.7vh" }} />
        </IconButton>
        <Box
          data-testid="dff-k-value"
          sx={{ minWidth: "1.4vh", textAlign: "center", color: "text.primary" }}
        >
          {currentK}
        </Box>
        <IconButton
          size="small"
          sx={{ padding: "0.2vh" }}
          aria-label={t("dff.increase")}
          disabled={currentK >= MAX_DFF_K || pending}
          onClick={() => changeK(1)}
        >
          <AddIcon sx={{ fontSize: "1.7vh" }} />
        </IconButton>
      </Box>

      {pending && (
        <Box
          data-testid="dff-computing"
          sx={{
            display: "flex",
            alignItems: "center",
            gap: "0.5vw",
            pl: "5.2vh",
            py: "0.4vh",
            fontSize: "1.2vh",
            color: "text.secondary",
          }}
        >
          <CircularProgress size={12} />
          <span>{t("dff.computing")}</span>
        </Box>
      )}

      {!pending && dff && conceptCount === 0 && (
        <Box
          data-testid="dff-empty"
          sx={{
            pl: "5.2vh",
            py: "0.4vh",
            fontSize: "1.2vh",
            color: "text.secondary",
          }}
        >
          {t("dff.empty")}
        </Box>
      )}

      {!pending && conceptCount > 0 && (
        <>
          {/* All parts — the full per-token argmax segmentation. */}
          <Box
            role="button"
            aria-pressed={activeConcept === undefined}
            data-testid="dff-concept-all"
            onClick={() => {
              setActiveResultKey(resultKey);
              setDffActiveConcept(resultKey, null);
            }}
            sx={rowSx(activeConcept === undefined)}
          >
            <Tooltip title={t("dff.allTooltip")}>
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.4vw",
                  width: "100%",
                }}
              >
                <Box
                  sx={{
                    width: "1.6vh",
                    height: "1.6vh",
                    borderRadius: "0.3vh",
                    flexShrink: 0,
                    background:
                      "conic-gradient(#e53935, #1e88e5, #43a047, #fb8c00)",
                  }}
                />
                <Box sx={{ flex: 1 }}>{t("dff.all")}</Box>
              </Box>
            </Tooltip>
          </Box>

          {/* One row per concept — isolate that part across all seeds. */}
          {Array.from({ length: conceptCount }, (_, k) => {
            const on = activeConcept === k;
            return (
              <Box
                key={k}
                role="button"
                aria-pressed={on}
                data-testid={`dff-concept-${k}`}
                onClick={() => {
                  setActiveResultKey(resultKey);
                  setDffActiveConcept(resultKey, k);
                }}
                sx={rowSx(on)}
              >
                <Tooltip title={t("dff.conceptTooltip")}>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.4vw",
                      width: "100%",
                    }}
                  >
                    <Box
                      sx={{
                        width: "1.6vh",
                        height: "1.6vh",
                        borderRadius: "0.3vh",
                        flexShrink: 0,
                        backgroundColor: conceptColorCss(k),
                        border: `1.5px solid ${conceptColorCss(k)}`,
                      }}
                    />
                    <Box sx={{ flex: 1 }}>{t("dff.concept", { n: k + 1 })}</Box>
                  </Box>
                </Tooltip>
              </Box>
            );
          })}

          {/* Per-species batch summary — how many seeds fed each factorization. */}
          {dff?.groups.map((g) => (
            <Box
              key={g.species}
              data-testid="dff-group"
              sx={{
                pl: "5.2vh",
                pr: "0.8vh",
                py: "0.2vh",
                fontSize: "1.05vh",
                color: "text.disabled",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {t("dff.group", { species: g.species, count: g.boxes.length })}
            </Box>
          ))}
        </>
      )}
    </Box>
  );
};

export default ConceptToggles;
