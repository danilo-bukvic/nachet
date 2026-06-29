import { Box, IconButton, Tooltip } from "@mui/material";
import LayersIcon from "@mui/icons-material/Layers";
import GradientIcon from "@mui/icons-material/Gradient";
import type { InferenceResult } from "@common/types";
import { useInferenceStore } from "@stores/useInferenceStore";
import { conceptColorCss } from "@common/dffColors";

/**
 * Per-concept DFF controls shown under a run in the Images panel.
 *
 * Each concept row has two toggles, for two mutually-exclusive modes:
 *  - Stack (Layers): adds this concept to the colored overlay; several concepts
 *    can be on at once (each cell shows its dominant active concept's color).
 *  - Heatmap (Gradient): shows ONLY this concept as a jet (blue→red) heatmap;
 *    single-select, and turning it on clears the colored stack.
 */

interface Props {
  /** "imageIndex:modelConfigId" for this run. */
  resultKey: string;
  result: InferenceResult;
}

const ConceptLayerToggles = ({ resultKey, result }: Props) => {
  const dffResults = useInferenceStore((s) => s.dffResults);
  const dffConcepts = useInferenceStore((s) => s.dffConcepts);
  const dffJet = useInferenceStore((s) => s.dffJet);
  const toggleDffConcept = useInferenceStore((s) => s.toggleDffConcept);
  const toggleDffJet = useInferenceStore((s) => s.toggleDffJet);
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

  const stack = dffConcepts.get(resultKey) ?? new Set<number>();
  const jet = dffJet.get(resultKey);

  return (
    <Box>
      {Array.from({ length: conceptCount }, (_, k) => {
        const stackOn = stack.has(k);
        const jetOn = jet === k;
        return (
          <Box
            key={k}
            data-testid={`concept-row-${k}`}
            sx={{
              display: "flex",
              alignItems: "center",
              gap: "0.4vw",
              pl: "5.2vh",
              pr: "0.8vh",
              py: "0.3vh",
              fontSize: "1.25vh",
              color: stackOn || jetOn ? "text.primary" : "text.secondary",
              backgroundColor: stackOn || jetOn ? "#E3F2FD" : "transparent",
              borderTop: "1px solid #f5f5f5",
            }}
          >
            {/* concept color swatch */}
            <Box
              sx={{
                width: "1.4vh",
                height: "1.4vh",
                borderRadius: "0.3vh",
                flexShrink: 0,
                backgroundColor: stackOn ? conceptColorCss(k) : "transparent",
                border: `1.5px solid ${conceptColorCss(k)}`,
              }}
            />
            <Box sx={{ flex: 1 }}>{`Concept ${k}`}</Box>

            {/* Stack (colored overlay, multi-select) */}
            <Tooltip title="Add to colored overlay" placement="top">
              <IconButton
                size="small"
                sx={{ padding: "0.2vh" }}
                aria-label={`overlay concept ${k}`}
                aria-pressed={stackOn}
                data-testid={`concept-stack-${k}`}
                onClick={() => {
                  setActiveResultKey(resultKey);
                  toggleDffConcept(resultKey, k);
                }}
              >
                <LayersIcon
                  sx={{
                    fontSize: "1.9vh",
                    color: stackOn ? "#1565c0" : "#bdbdbd",
                  }}
                />
              </IconButton>
            </Tooltip>

            {/* Jet heatmap (single concept, single-select) */}
            <Tooltip title="Show as heatmap (single)" placement="top">
              <IconButton
                size="small"
                sx={{ padding: "0.2vh" }}
                aria-label={`heatmap concept ${k}`}
                aria-pressed={jetOn}
                data-testid={`concept-jet-${k}`}
                onClick={() => {
                  setActiveResultKey(resultKey);
                  toggleDffJet(resultKey, k);
                }}
              >
                <GradientIcon
                  sx={{
                    fontSize: "1.9vh",
                    color: jetOn ? "#d32f2f" : "#bdbdbd",
                  }}
                />
              </IconButton>
            </Tooltip>
          </Box>
        );
      })}
    </Box>
  );
};

export default ConceptLayerToggles;
