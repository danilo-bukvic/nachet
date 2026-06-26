import { useEffect, useRef, useState } from "react";
import { Box, Collapse } from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import GrainIcon from "@mui/icons-material/Grain";
import type { InferenceBox, InferenceResult } from "@common/types";
import {
  useInferenceStore,
  type DffBoxResult,
} from "@stores/useInferenceStore";

/**
 * Per-seed DFF concept cutouts shown under a run in the Images panel.
 *
 * For each detected seed that has DFF data, renders an expandable sub-row whose
 * body shows the notebook layout: the seed cutout repeated K times, each with a
 * different concept's jet heatmap overlaid.
 */

const THUMB = 78; // px

// "jet" colormap (matches the notebook's cmap="jet"): value [0,1] -> RGB.
const jetColor = (t: number): [number, number, number] => {
  const v = Math.max(0, Math.min(1, t));
  const clamp = (x: number) => Math.max(0, Math.min(1, x));
  return [
    Math.round(clamp(1.5 - Math.abs(4 * v - 3)) * 255),
    Math.round(clamp(1.5 - Math.abs(4 * v - 2)) * 255),
    Math.round(clamp(1.5 - Math.abs(4 * v - 1)) * 255),
  ];
};

/** Load an HTMLImageElement for a src (browser-cached across thumbs). */
const useImageElement = (src: string): HTMLImageElement | null => {
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    let alive = true;
    const el = new Image();
    el.onload = () => {
      if (alive) setImg(el);
    };
    el.src = src;
    return () => {
      alive = false;
    };
  }, [src]);
  return img;
};

interface ThumbProps {
  img: HTMLImageElement | null;
  box: InferenceBox;
  dff: DffBoxResult;
  concept: number;
}

/** One seed cutout with a single concept's jet heatmap overlaid (~0.55 alpha). */
const ConceptThumb = ({ img, box, dff, concept }: ThumbProps) => {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const sx = Math.max(0, box.topX);
    const sy = Math.max(0, box.topY);
    const sw = Math.max(1, box.bottomX - box.topX);
    const sh = Math.max(1, box.bottomY - box.topY);

    ctx.clearRect(0, 0, THUMB, THUMB);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, THUMB, THUMB);

    const g = dff.grid;
    const heat = dff.heatmaps[concept];
    if (heat && g * g === heat.length) {
      const small = document.createElement("canvas");
      small.width = g;
      small.height = g;
      const sctx = small.getContext("2d");
      if (sctx) {
        const id = sctx.createImageData(g, g);
        for (let p = 0; p < g * g; p++) {
          const [r, gg, b] = jetColor(heat[p]);
          const o = p * 4;
          id.data[o] = r;
          id.data[o + 1] = gg;
          id.data[o + 2] = b;
          id.data[o + 3] = 140; // ~0.55 alpha, like the notebook
        }
        sctx.putImageData(id, 0, 0);
        ctx.drawImage(small, 0, 0, THUMB, THUMB);
      }
    }
  }, [img, box, dff, concept]);

  return (
    <canvas
      ref={ref}
      width={THUMB}
      height={THUMB}
      style={{ borderRadius: 4, display: "block" }}
    />
  );
};

interface Props {
  imageSrc: string;
  /** "imageIndex:modelConfigId" — prefix for this run's dff keys. */
  resultKey: string;
  result: InferenceResult;
}

const SeedConceptCutouts = ({ imageSrc, resultKey, result }: Props) => {
  const dffResults = useInferenceStore((s) => s.dffResults);
  const img = useImageElement(imageSrc);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const seeds = result.boxes
    .map((box, i) => ({
      box,
      i,
      label: result.classifications[i] || box.label || `Seed ${i + 1}`,
      dff: dffResults.get(`${resultKey}:${box.boxId}`),
    }))
    .filter((s): s is typeof s & { dff: DffBoxResult } => Boolean(s.dff));

  if (seeds.length === 0) return null;

  const toggle = (boxId: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(boxId)) next.delete(boxId);
      else next.add(boxId);
      return next;
    });

  return (
    <Box>
      {seeds.map(({ box, i, label, dff }) => {
        const open = expanded.has(box.boxId);
        return (
          <Box key={box.boxId}>
            {/* Seed sub-heading */}
            <Box
              role="button"
              aria-expanded={open}
              data-testid={`seed-concepts-row-${i}`}
              onClick={() => toggle(box.boxId)}
              sx={{
                display: "flex",
                alignItems: "center",
                gap: "0.3vw",
                pl: "5.2vh",
                pr: "0.8vh",
                py: "0.35vh",
                fontSize: "1.25vh",
                cursor: "pointer",
                color: "text.secondary",
                borderTop: "1px solid #f5f5f5",
                "&:hover": { backgroundColor: "#F5F5F5" },
              }}
            >
              {open ? (
                <ExpandMoreIcon sx={{ fontSize: "1.8vh", color: "#7b1fa2" }} />
              ) : (
                <ChevronRightIcon
                  sx={{ fontSize: "1.8vh", color: "#7b1fa2" }}
                />
              )}
              <GrainIcon sx={{ fontSize: "1.6vh", color: "#7b1fa2" }} />
              <Box
                sx={{
                  flex: 1,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {`${i + 1}. ${label}`}
              </Box>
            </Box>

            {/* Concept cutouts (one per concept) */}
            <Collapse in={open} unmountOnExit>
              <Box
                sx={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: "0.5vh",
                  pl: "5.2vh",
                  pr: "0.8vh",
                  py: "0.5vh",
                }}
              >
                {dff.heatmaps.map((_, k) => (
                  <Box key={k} sx={{ textAlign: "center" }}>
                    <ConceptThumb img={img} box={box} dff={dff} concept={k} />
                    <Box sx={{ fontSize: "1.05vh", color: "text.disabled" }}>
                      {`concept ${k}`}
                    </Box>
                  </Box>
                ))}
              </Box>
            </Collapse>
          </Box>
        );
      })}
    </Box>
  );
};

export default SeedConceptCutouts;
