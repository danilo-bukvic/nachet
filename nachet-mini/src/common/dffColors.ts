// Colors for DFF concept overlays + their toggle swatches, so the on-image
// segmentation for "Concept k" matches the color shown in the toggle list.
//
// These are matplotlib's `tab10` / `tab20` qualitative colormaps. Qualitative
// (not sequential) is the right family here: a concept index is a *category*,
// not a magnitude, so the palette's job is to make neighbouring concepts
// instantly tellable apart rather than to imply any ordering or strength.
// Concept presence is drawn as flat color and absence as nothing, so opacity
// never encodes a value — see InferenceOverlay.
//
// https://matplotlib.org/stable/users/explain/colors/colormaps.html

/** matplotlib `tab10` — 10 maximally distinct categorical colors. */
export const TAB10 = [
  "#1f77b4", // blue
  "#ff7f0e", // orange
  "#2ca02c", // green
  "#d62728", // red
  "#9467bd", // purple
  "#8c564b", // brown
  "#e377c2", // pink
  "#7f7f7f", // gray
  "#bcbd22", // olive
  "#17becf", // cyan
] as const;

/** matplotlib `tab20` — the tab10 hues, each paired with a lighter tint. */
export const TAB20 = [
  "#1f77b4",
  "#aec7e8",
  "#ff7f0e",
  "#ffbb78",
  "#2ca02c",
  "#98df8a",
  "#d62728",
  "#ff9896",
  "#9467bd",
  "#c5b0d5",
  "#8c564b",
  "#c49c94",
  "#e377c2",
  "#f7b6d2",
  "#7f7f7f",
  "#c7c7c7",
  "#bcbd22",
  "#dbdb8d",
  "#17becf",
  "#9edae5",
] as const;

/**
 * Pick the colormap for a given number of concepts: `tab10` while the concepts
 * fit in it, `tab20` beyond. Staying on tab10 as long as possible keeps the
 * colors as far apart as they can be — tab20's tints are much closer together.
 */
export const conceptPalette = (conceptCount: number): readonly string[] =>
  conceptCount > TAB10.length ? TAB20 : TAB10;

/** Hex color for a concept index (wraps if there are more concepts than colors). */
export const conceptColorHex = (
  concept: number,
  conceptCount: number = TAB10.length,
): string => {
  const palette = conceptPalette(conceptCount);
  return palette[concept % palette.length];
};

const hexToRgb = (hex: string): [number, number, number] => {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
};

/** RGB triple for a concept index (for canvas pixel writes). */
export const conceptColorRgb = (
  concept: number,
  conceptCount: number = TAB10.length,
): [number, number, number] => hexToRgb(conceptColorHex(concept, conceptCount));

/** CSS color for a concept index (for the toggle swatches). */
export const conceptColorCss = (
  concept: number,
  conceptCount: number = TAB10.length,
): string => conceptColorHex(concept, conceptCount);
