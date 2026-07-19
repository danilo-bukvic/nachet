import { Box, ToggleButton, ToggleButtonGroup, Tooltip } from "@mui/material";
import { useTranslation } from "react-i18next";
import type { InferenceResult } from "@common/types";
import { useInferenceStore } from "@stores/useInferenceStore";
import CamRankToggles from "@components/CamRankToggles";
import ConceptToggles from "@components/ConceptToggles";

/**
 * Explainability controls shown under a run in the Images panel, with a CAM/DFF
 * mode switch in the run header. CAM (per-species class activation) is the
 * default; DFF (feature factorization into shared parts) is the second mode.
 * Both read the patched model's `swin_layernorm` output, so a single capability
 * gate — CAM data existing for the run — enables the whole control.
 */

interface Props {
  /** "imageIndex:modelConfigId" for this run. */
  resultKey: string;
  result: InferenceResult;
}

const ExplainToggles = ({ resultKey, result }: Props) => {
  const { t } = useTranslation("main");
  const camResults = useInferenceStore((s) => s.camResults);
  const explainMode = useInferenceStore((s) => s.explainMode);
  const setExplainMode = useInferenceStore((s) => s.setExplainMode);
  const setActiveResultKey = useInferenceStore((s) => s.setActiveResultKey);

  let capable = false;
  for (const box of result.boxes) {
    if (camResults.has(`${resultKey}:${box.boxId}`)) {
      capable = true;
      break;
    }
  }
  if (!capable) return null;

  const mode = explainMode.get(resultKey) ?? "cam";

  return (
    <Box data-testid="explain-toggles">
      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          py: "0.3vh",
          borderTop: "1px solid #f5f5f5",
        }}
      >
        <ToggleButtonGroup
          size="small"
          exclusive
          value={mode}
          aria-label={t("explain.modeLabel")}
          onChange={(_, next: string | null) => {
            if (!next) return; // keep one mode always selected
            setActiveResultKey(resultKey);
            setExplainMode(resultKey, next as "cam" | "dff");
          }}
        >
          <ToggleButton
            value="cam"
            data-testid="explain-mode-cam"
            sx={{ px: "1vh", py: "0.1vh", fontSize: "1.1vh", lineHeight: 1.2 }}
          >
            <Tooltip title={t("explain.camTooltip")}>
              <span>{t("explain.cam")}</span>
            </Tooltip>
          </ToggleButton>
          <ToggleButton
            value="dff"
            data-testid="explain-mode-dff"
            sx={{ px: "1vh", py: "0.1vh", fontSize: "1.1vh", lineHeight: 1.2 }}
          >
            <Tooltip title={t("explain.dffTooltip")}>
              <span>{t("explain.dff")}</span>
            </Tooltip>
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>
      {mode === "cam" ? (
        <CamRankToggles resultKey={resultKey} result={result} />
      ) : (
        <ConceptToggles resultKey={resultKey} />
      )}
    </Box>
  );
};

export default ExplainToggles;
