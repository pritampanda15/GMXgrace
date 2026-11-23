"""Parser for GROMACS .xpm 2D matrix files (FEL, DSSP, contact maps, RMSD/covariance matrices)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from gmxgrace.grace_text import grace_to_matplotlib
from gmxgrace.models import MatrixData, ParsedDataset
from gmxgrace.parsers.xvg import detect_unit

_META_COMMENT_RE = re.compile(r"/\*\s*([a-zA-Z0-9_-]+)\s*:\s*(.*?)\s*\*/")
_QUOTED_LINE_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_COLOR_DEF_RE = re.compile(
    r'^(?P<code>.+?)\s+c\s+(?P<color>#[0-9A-Fa-f]{3,8}|[A-Za-z0-9_]+)\s*(?:/\*\s*"(?P<label>[^"]*)"\s*\*/)?'
)


def parse_xpm_text(text: str, filename: str = "matrix.xpm") -> ParsedDataset:
    """Parse a GROMACS XPM matrix text into a ParsedDataset(kind='matrix')."""
    title = ""
    legend = ""
    x_label = ""
    y_label = ""
    xpm_type = ""
    x_axis_vals: list[float] = []
    y_axis_vals: list[float] = []

    # 1. Extract /* key: value */ metadata comments (x-axis and y-axis may span multiple lines)
    for line in text.splitlines():
        m = _META_COMMENT_RE.search(line)
        if not m:
            continue
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        if key == "title":
            title = val
        elif key == "legend":
            legend = val
        elif key == "x-label":
            x_label = val
        elif key == "y-label":
            y_label = val
        elif key == "type":
            xpm_type = val.lower()
        elif key == "x-axis":
            for tok in val.split():
                try:
                    x_axis_vals.append(float(tok))
                except ValueError:
                    pass
        elif key == "y-axis":
            for tok in val.split():
                try:
                    y_axis_vals.append(float(tok))
                except ValueError:
                    pass

    # 2. Extract C array quoted lines, preserving trailing /* "label" */ comments on color lines
    lines = text.splitlines()
    quoted_entries: list[tuple[str, str]] = []  # (quoted_content, full_line)
    in_array = False
    for line in lines:
        stripped = line.lstrip()
        if "{" in line and "static" in line:
            in_array = True
            continue
        if not in_array:
            continue
        if stripped.startswith("/*"):
            continue
        m = _QUOTED_LINE_RE.search(line)
        if m:
            quoted_entries.append((m.group(1), line))

    if not quoted_entries:
        raise ValueError(f"No XPM matrix data found in {filename}")

    # Header line: "<width> <height> <ncolors> <chars_per_pixel>"
    header_parts = quoted_entries[0][0].split()
    if len(header_parts) < 4:
        raise ValueError(f"Invalid XPM header in {filename}: {quoted_entries[0][0]!r}")
    width, height, ncolors, cpp = (int(header_parts[0]), int(header_parts[1]), int(header_parts[2]), int(header_parts[3]))

    color_entries = quoted_entries[1 : 1 + ncolors]
    pixel_entries = quoted_entries[1 + ncolors : 1 + ncolors + height]

    if len(pixel_entries) < height:
        raise ValueError(f"Expected {height} XPM pixel rows in {filename}, found {len(pixel_entries)}")

    # Parse color table
    code_to_color: dict[str, str] = {}
    code_to_label: dict[str, str] = {}
    ordered_codes: list[str] = []

    for content, full_line in color_entries:
        code = content[:cpp]
        rest = content[cpp:].strip()
        # Extract color token after 'c'
        c_match = re.search(r"\bc\s+(#[0-9A-Fa-f]{3,8}|[A-Za-z0-9_]+)", rest)
        color = c_match.group(1) if c_match else "#888888"
        # Extract label from trailing /* "label" */ comment on full_line
        lbl_match = re.search(r'/\*\s*"([^"]*)"\s*\*/', full_line)
        label = lbl_match.group(1).strip() if lbl_match else code
        code_to_color[code] = color
        code_to_label[code] = label
        ordered_codes.append(code)

    # Determine whether values are continuous (numeric) or discrete (categorical e.g., DSSP)
    numeric_map: dict[str, float] = {}
    all_numeric = True
    for code in ordered_codes:
        lbl = code_to_label[code]
        try:
            numeric_map[code] = float(lbl)
        except ValueError:
            all_numeric = False
            break

    is_discrete = (not all_numeric) or (xpm_type == "discrete" and not all_numeric)

    categories: list[str] = []
    category_colors: list[str] = []
    code_to_val: dict[str, float] = {}

    if is_discrete:
        for idx, code in enumerate(ordered_codes):
            code_to_val[code] = float(idx)
            categories.append(code_to_label[code])
            category_colors.append(code_to_color[code])
    else:
        code_to_val = numeric_map

    # Build 2D matrix.
    # NOTE: In GROMACS XPM files, the FIRST pixel row corresponds to the TOP (highest y-axis value),
    # and the LAST pixel row corresponds to the BOTTOM (lowest y-axis value).
    # We flip vertically so matrix[0, :] aligns with y_axis[0] (bottom) for origin='lower' plotting.
    mat = np.full((height, width), np.nan, dtype=float)
    for row_idx, (row_str, _) in enumerate(pixel_entries):
        target_row = height - 1 - row_idx
        for col_idx in range(width):
            start = col_idx * cpp
            token = row_str[start : start + cpp]
            if token in code_to_val:
                mat[target_row, col_idx] = code_to_val[token]

    # Axis coordinates
    if len(x_axis_vals) == width:
        x_axis = np.asarray(x_axis_vals, dtype=float)
    elif len(x_axis_vals) == width + 1:
        arr = np.asarray(x_axis_vals, dtype=float)
        x_axis = 0.5 * (arr[:-1] + arr[1:])
    elif len(x_axis_vals) >= 2:
        x_axis = np.linspace(x_axis_vals[0], x_axis_vals[-1], width)
    else:
        x_axis = np.arange(width, dtype=float)

    if len(y_axis_vals) == height:
        y_axis = np.asarray(y_axis_vals, dtype=float)
    elif len(y_axis_vals) == height + 1:
        arr = np.asarray(y_axis_vals, dtype=float)
        y_axis = 0.5 * (arr[:-1] + arr[1:])
    elif len(y_axis_vals) >= 2:
        y_axis = np.linspace(y_axis_vals[0], y_axis_vals[-1], height)
    else:
        y_axis = np.arange(height, dtype=float)

    # Detect GROMACS matrix subtype
    low = f"{filename} {title} {legend}".lower()
    if "gibbs" in low or "free energy" in low or "fel" in low or "sham" in low:
        detected = "fel"
    elif "secondary structure" in low or "dssp" in low or is_discrete:
        detected = "dssp"
    elif "contact" in low or "dist" in low:
        detected = "contact_map"
    elif "covar" in low or "corr" in low:
        detected = "covariance"
    else:
        detected = "matrix"

    x_clean = grace_to_matplotlib(x_label) or "X"
    y_clean = grace_to_matplotlib(y_label) or "Y"
    legend_clean = grace_to_matplotlib(legend) or ("State" if is_discrete else "Value")

    return ParsedDataset(
        filename=filename,
        file_format="xpm",
        kind="matrix",
        title=grace_to_matplotlib(title) or Path(filename).stem,
        x_label=x_clean,
        y_label=y_clean,
        x_unit=detect_unit(x_label) or detect_unit(x_clean),
        y_unit=detect_unit(y_label) or detect_unit(y_clean),
        detected_type=detected,
        matrix=MatrixData(
            values=mat,
            x_axis=x_axis,
            y_axis=y_axis,
            is_discrete=is_discrete,
            categories=categories,
            category_colors=category_colors,
            legend_label=legend_clean,
            raw_colors=code_to_color,
        ),
        metadata={"width": width, "height": height, "ncolors": ncolors, "cpp": cpp},
    )


def parse_xpm(path: str | Path) -> ParsedDataset:
    """Parse a GROMACS .xpm file from disk."""
    p = Path(path)
    return parse_xpm_text(p.read_text(encoding="utf-8", errors="replace"), filename=p.name)
