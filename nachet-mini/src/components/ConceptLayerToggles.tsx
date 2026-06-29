import { Box } from "@mui/material";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffOutlinedIcon from "@mui/icons-material/VisibilityOffOutlined";
import type { InferenceResult } from "@common/types";
import { useInferenceStore } from "@stores/useInferenceStore";
import { conceptColorCss } from "@common/dffColors";

/**
 * Per-concept DFF toggles shown under a run in the Images panel.
 *
 * Lists one row per concept ("Concept 0", "Concept 1", ...). Toggling a concept
 * overlays its heatmap on every detected seed of this run in the image viewer;
 * several concepts can be active at once (their colors blend on the image).
 */

interface Props {
  /** "imageIndex:modelConfigId" for this run. */
  resultKey: string;
  result: InferenceResult;
}

const ConceptLayerToggles = ({ resultKey, result }: Props) => {
  const dffResults = useInferenceStore((s) => s.dffResults);
  const dffConcepts = useInferenceStore((s) => s.dffConcepts);
  const toggleDffConcept = useInferenceStore((s) => s.toggleDffConcept);
  const setActiveResultKey = useInferenceStore((s) => s.setActiveResultKey);

  // Number of concepts = heatmap count from the first box that has DFF data.
  let conceptCount = 0;
  for (const box of result.boxes) {
    const dff = dffResults.get(`${resultKey}:${box.boxId}`);
    if (dff) {
      conceptCount = dff.heatmaps.length;
      break;
    }
  }
  if (conceptCount === 0) return null;

  const active = dffConcepts.get(resultKey) ?? new Set<number>();

  return (
    <Box>
      {Array.from({ length: conceptCount }, (_, k) => {
        const on = active.has(k);
        return (
          <Box
            key={k}
            role="button"
            aria-pressed={on}
            data-testid={`concept-toggle-${k}`}
            onClick={() => {
              setActiveResultKey(resultKey);
              toggleDffConcept(resultKey, k);
            }}
            sx={{
              display: "flex",
              alignItems: "center",
              gap: "0.4vw",
              pl: "5.2vh",
              pr: "0.8vh",
              py: "0.4vh",
              fontSize: "1.25vh",
              cursor: "pointer",
              color: on ? "text.primary" : "text.secondary",
              backgroundColor: on ? "#E3F2FD" : "transparent",
              borderTop: "1px solid #f5f5f5",
              "&:hover": { backgroundColor: on ? "#E3F2FD" : "#F5F5F5" },
            }}
          >
            {/* concept color swatch */}
            <Box
              sx={{
                width: "1.4vh",
                height: "1.4vh",
                borderRadius: "0.3vh",
                flexShrink: 0,
                backgroundColor: on ? conceptColorCss(k) : "transparent",
                border: `1.5px solid ${conceptColorCss(k)}`,
              }}
            />
            <Box sx={{ flex: 1 }}>{`Concept ${k}`}</Box>
            {on ? (
              <VisibilityIcon sx={{ fontSize: "1.8vh", color: "#1565c0" }} />
            ) : (
              <VisibilityOffOutlinedIcon
                sx={{ fontSize: "1.8vh", color: "#bdbdbd" }}
              />
            )}
          </Box>
        );
      })}
    </Box>
  );
};

export default ConceptLayerToggles;
