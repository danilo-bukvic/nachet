import { describe, it, expect } from "vitest";
import {
  TAB10,
  TAB20,
  conceptPalette,
  conceptColorHex,
  conceptColorRgb,
  conceptColorCss,
} from "../dffColors";

describe("dffColors", () => {
  it("uses matplotlib tab10 by default, in order", () => {
    expect(conceptColorHex(0)).toBe("#1f77b4"); // blue, not red
    expect(conceptColorHex(1)).toBe("#ff7f0e");
    expect(conceptColorHex(2)).toBe("#2ca02c");
    expect(conceptColorHex(3)).toBe("#d62728");
  });

  it("stays on tab10 while the concepts fit, then switches to tab20", () => {
    expect(conceptPalette(1)).toBe(TAB10);
    expect(conceptPalette(10)).toBe(TAB10);
    expect(conceptPalette(11)).toBe(TAB20);
    expect(conceptColorHex(1, 11)).toBe(TAB20[1]);
  });

  it("wraps around when there are more concepts than colors", () => {
    expect(conceptColorHex(TAB10.length)).toBe(TAB10[0]);
    expect(conceptColorHex(TAB10.length + 2)).toBe(TAB10[2]);
  });

  it("converts to an RGB triple for canvas writes", () => {
    expect(conceptColorRgb(0)).toEqual([0x1f, 0x77, 0xb4]);
    expect(conceptColorRgb(3)).toEqual([0xd6, 0x27, 0x28]);
  });

  it("exposes the same color to swatches as to the overlay", () => {
    for (let k = 0; k < TAB10.length; k++) {
      const [r, g, b] = conceptColorRgb(k);
      expect(conceptColorCss(k)).toBe(conceptColorHex(k));
      expect(
        `#${[r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("")}`,
      ).toBe(conceptColorHex(k));
    }
  });
});
