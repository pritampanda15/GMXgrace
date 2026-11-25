"""Publication journal presets and colorblind-safe scientific palettes for gmxgrace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class JournalPreset:
    """Publication layout and typography preset matching major scientific journals."""

    id: str
    name: str
    description: str
    widths_mm: dict[str, float]  # single, one_half, double
    font_family: str
    font_size: float
    title_size: float
    label_size: float
    tick_size: float
    legend_size: float
    panel_label_size: float
    line_width: float
    axes_linewidth: float
    tick_direction: str  # "out" | "in"
    top_right_spines: bool
    tick_top_right: bool
    default_palette: str
    default_panel_labels: str  # lowercase | uppercase


JOURNAL_PRESETS: dict[str, JournalPreset] = {
    "nature": JournalPreset(
        id="nature",
        name="Nature Portfolio",
        description="Nature / Nature Methods / Nature Comm (89 mm / 183 mm, sans-serif, outward ticks, open top/right)",
        widths_mm={"single": 89.0, "one_half": 120.0, "double": 183.0},
        font_family="sans-serif",
        font_size=7.5,
        title_size=8.5,
        label_size=8.0,
        tick_size=7.0,
        legend_size=7.0,
        panel_label_size=9.5,
        line_width=1.15,
        axes_linewidth=0.75,
        tick_direction="out",
        top_right_spines=False,
        tick_top_right=False,
        default_palette="nature-npg",
        default_panel_labels="lowercase",
    ),
    "science": JournalPreset(
        id="science",
        name="Science (AAAS)",
        description="Science / Science Advances (55 mm / 120 mm / 175 mm, crisp sans-serif, outward ticks)",
        widths_mm={"single": 75.0, "one_half": 120.0, "double": 175.0},
        font_family="sans-serif",
        font_size=7.5,
        title_size=8.5,
        label_size=8.0,
        tick_size=7.0,
        legend_size=6.8,
        panel_label_size=9.5,
        line_width=1.1,
        axes_linewidth=0.75,
        tick_direction="out",
        top_right_spines=False,
        tick_top_right=False,
        default_palette="okabe-ito",
        default_panel_labels="uppercase",
    ),
    "acs": JournalPreset(
        id="acs",
        name="ACS (JCTC / JACS / JCIM)",
        description="American Chemical Society (84.6 mm / 177.8 mm, 4-spine boxed frame, inward ticks)",
        widths_mm={"single": 84.6, "one_half": 130.0, "double": 177.8},
        font_family="sans-serif",
        font_size=8.0,
        title_size=9.0,
        label_size=8.5,
        tick_size=7.5,
        legend_size=7.5,
        panel_label_size=9.5,
        line_width=1.25,
        axes_linewidth=0.85,
        tick_direction="in",
        top_right_spines=True,
        tick_top_right=True,
        default_palette="acs-classic",
        default_panel_labels="parens_lower",
    ),
    "biophysj": JournalPreset(
        id="biophysj",
        name="Biophysical Journal",
        description="Biophysical Journal / Cell Press (85 mm / 178 mm, 4-spine frame, inward ticks, high contrast)",
        widths_mm={"single": 85.0, "one_half": 128.0, "double": 178.0},
        font_family="sans-serif",
        font_size=8.0,
        title_size=9.0,
        label_size=8.5,
        tick_size=7.5,
        legend_size=7.5,
        panel_label_size=10.0,
        line_width=1.3,
        axes_linewidth=0.9,
        tick_direction="in",
        top_right_spines=True,
        tick_top_right=True,
        default_palette="okabe-ito",
        default_panel_labels="uppercase",
    ),
    "pnas": JournalPreset(
        id="pnas",
        name="PNAS",
        description="Proceedings of the National Academy of Sciences (87 mm / 114 mm / 178 mm, clean open frame)",
        widths_mm={"single": 87.0, "one_half": 114.0, "double": 178.0},
        font_family="sans-serif",
        font_size=8.0,
        title_size=9.0,
        label_size=8.5,
        tick_size=7.5,
        legend_size=7.5,
        panel_label_size=9.5,
        line_width=1.2,
        axes_linewidth=0.8,
        tick_direction="out",
        top_right_spines=False,
        tick_top_right=False,
        default_palette="tol-bright",
        default_panel_labels="uppercase",
    ),
    "grace": JournalPreset(
        id="grace",
        name="Modern xmgrace",
        description="Classic GROMACS xmgrace feel modernized for publication (boxed frame, bold inward ticks)",
        widths_mm={"single": 110.0, "one_half": 145.0, "double": 185.0},
        font_family="sans-serif",
        font_size=9.0,
        title_size=10.5,
        label_size=9.5,
        tick_size=8.5,
        legend_size=8.5,
        panel_label_size=11.0,
        line_width=1.4,
        axes_linewidth=1.0,
        tick_direction="in",
        top_right_spines=True,
        tick_top_right=True,
        default_palette="okabe-ito",
        default_panel_labels="parens_upper",
    ),
}

PALETTES: dict[str, list[str]] = {
    "okabe-ito": [
        "#0072B2", "#D55E00", "#009E73", "#CC79A7",
        "#E69F00", "#56B4E9", "#000000", "#F0E442",
    ],
    "nature-npg": [
        "#3C5488", "#E64B35", "#00A087", "#4DBBD5",
        "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
    ],
    "acs-classic": [
        "#1B4F72", "#C0392B", "#1E8449", "#D68910",
        "#6C3483", "#2E86C1", "#7F8C8D", "#117A65",
    ],
    "tol-bright": [
        "#4477AA", "#EE6677", "#228833", "#CCBB44",
        "#66CCEE", "#AA3377", "#BBBBBB", "#000000",
    ],
    "tol-muted": [
        "#332288", "#88CCEE", "#44AA99", "#117733",
        "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499",
    ],
    "viridis-discrete": [
        "#440154", "#3B528B", "#21918C", "#5EC962", "#D6B717", "#FDE725",
    ],
    "grayscale-pub": [
        "#111111", "#555555", "#888888", "#222222", "#666666", "#AAAAAA",
    ],
}

SEQUENTIAL_CMAPS = [
    "RdYlBu_r",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "coolwarm",
    "Spectral_r",
    "turbo",
    "Blues",
    "YlGnBu",
    "cubehelix_r",
]


def get_journal_preset(journal_id: str) -> JournalPreset:
    return JOURNAL_PRESETS.get(journal_id, JOURNAL_PRESETS["nature"])


def get_palette(palette_id: str) -> list[str]:
    return PALETTES.get(palette_id, PALETTES["okabe-ito"])


def build_rc_params(
    preset: JournalPreset,
    *,
    font_family: str | None = None,
    font_size: float | None = None,
    line_width: float | None = None,
) -> dict[str, Any]:
    """Build Matplotlib rcParams dictionary for thread-safe rc_context rendering."""
    fam = font_family or preset.font_family
    scale = (font_size / preset.font_size) if font_size else 1.0
    lw = line_width if line_width is not None else preset.line_width

    return {
        "font.family": fam,
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
        "font.serif": ["Times New Roman", "Georgia", "DejaVu Serif"],
        "font.size": preset.font_size * scale,
        "axes.titlesize": preset.title_size * scale,
        "axes.titleweight": "semibold",
        "axes.labelsize": preset.label_size * scale,
        "axes.linewidth": preset.axes_linewidth,
        "axes.spines.top": preset.top_right_spines,
        "axes.spines.right": preset.top_right_spines,
        "axes.edgecolor": "#1F2937",
        "axes.labelcolor": "#111827",
        "text.color": "#111827",
        "xtick.labelsize": preset.tick_size * scale,
        "ytick.labelsize": preset.tick_size * scale,
        "xtick.direction": preset.tick_direction,
        "ytick.direction": preset.tick_direction,
        "xtick.top": preset.tick_top_right,
        "ytick.right": preset.tick_top_right,
        "xtick.major.width": preset.axes_linewidth,
        "ytick.major.width": preset.axes_linewidth,
        "xtick.minor.width": preset.axes_linewidth * 0.75,
        "ytick.minor.width": preset.axes_linewidth * 0.75,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.color": "#1F2937",
        "ytick.color": "#1F2937",
        "legend.fontsize": preset.legend_size * scale,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.5,
        "lines.linewidth": lw,
        "grid.color": "#CBD5E1",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.65,
        # Editable vector typography for Adobe Illustrator / Inkscape / journals
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "path.simplify": True,
        "path.simplify_threshold": 0.2,
        "mathtext.default": "regular",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }


def list_presets_metadata() -> dict[str, Any]:
    """Return serializable metadata of all journals, palettes, and colormaps for the Web UI."""
    return {
        "journals": [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "widths_mm": p.widths_mm,
                "font_size": p.font_size,
                "line_width": p.line_width,
                "default_palette": p.default_palette,
                "default_panel_labels": p.default_panel_labels,
            }
            for p in JOURNAL_PRESETS.values()
        ],
        "palettes": PALETTES,
        "colormaps": SEQUENTIAL_CMAPS,
    }
