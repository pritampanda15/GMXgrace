"""Parser for GROMACS / Grace .xvg files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TextIO

import numpy as np

from gmxgrace.grace_text import grace_to_matplotlib
from gmxgrace.models import ParsedDataset, SeriesData

_QUOTED_RE = re.compile(r'"(.*)"')
_LEGEND_RE = re.compile(r'^@\s*s(\d+)\s+legend\s+"(.*)"\s*$', re.IGNORECASE)
_TYPE_RE = re.compile(r"^@\s*TYPE\s+(\S+)", re.IGNORECASE)
_UNIT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\(\s*ps\s*\)", re.IGNORECASE), "ps"),
    (re.compile(r"\(\s*ns\s*\)", re.IGNORECASE), "ns"),
    (re.compile(r"\(\s*(?:us|µs|μs)\s*\)", re.IGNORECASE), "us"),
    (re.compile(r"\(\s*nm\s*(?:\^2|²|\\S2\\N)\s*\)", re.IGNORECASE), "nm2"),
    (re.compile(r"\(\s*(?:Å|A)\s*(?:\^2|²|\\S2\\N)\s*\)", re.IGNORECASE), "A2"),
    (re.compile(r"\(\s*nm\s*\)", re.IGNORECASE), "nm"),
    (re.compile(r"\(\s*(?:Å|A|angstrom)\s*\)", re.IGNORECASE), "A"),
    (re.compile(r"\(\s*kJ\s*/\s*mol\s*\)|\bkJ\s*mol[-⁻¹]", re.IGNORECASE), "kJ/mol"),
    (re.compile(r"\(\s*kcal\s*/\s*mol\s*\)|\bkcal\s*mol[-⁻¹]", re.IGNORECASE), "kcal/mol"),
]


def detect_unit(label: str) -> str | None:
    """Detect canonical physical unit from an axis label."""
    for pat, unit in _UNIT_PATTERNS:
        if pat.search(label):
            return unit
    return None


def detect_gromacs_type(
    filename: str,
    title: str,
    x_label: str,
    y_label: str,
    gmx_command: str,
) -> str:
    """Heuristically classify the GROMACS analysis output type."""
    haystack = f"{filename} {title} {x_label} {y_label} {gmx_command}".lower()
    if "rmsf" in haystack or "fluctuation" in haystack:
        return "rmsf"
    if "rmsd" in haystack or "gmx rms" in haystack:
        return "rmsd"
    if "gyrat" in haystack or "rg" in haystack.split() or "radius of gyration" in haystack:
        return "gyrate"
    if "sasa" in haystack or "solvent accessible" in haystack or "area" in haystack and "nm" in haystack:
        return "sasa"
    if "hbond" in haystack or "hydrogen bond" in haystack:
        return "hbond"
    if "wham" in haystack or "pmf" in haystack or "profile" in haystack and "umbrella" in haystack:
        return "pmf"
    if "histo" in haystack and ("wham" in haystack or "umbrella" in haystack or "pull" in haystack):
        return "wham_hist"
    if "energy" in haystack or "potential" in haystack or "temperature" in haystack or "pressure" in haystack:
        return "energy"
    if "mindist" in haystack or "distance" in haystack:
        return "distance"
    if "rama" in haystack or ("phi" in haystack and "psi" in haystack):
        return "ramachandran"
    if "pca" in haystack or "proj" in haystack or "eigvec" in haystack:
        return "pca"
    return "generic"


def _extract_quoted(line: str) -> str:
    m = _QUOTED_RE.search(line)
    return m.group(1) if m else ""


def parse_xvg_text(text: str, filename: str = "data.xvg", nxy_hint: bool = False) -> ParsedDataset:
    """Parse GROMACS .xvg text into a ParsedDataset."""
    title = ""
    subtitle = ""
    x_label = ""
    y_label = ""
    gmx_command = ""
    xy_type = "xy"
    legends: dict[int, str] = {}

    blocks: list[list[list[float]]] = []
    current_block: list[list[float]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "gmx " in line or "g_" in line or "command line" in line.lower():
                if not gmx_command:
                    gmx_command = line.lstrip("#").strip()
            continue
        if line.startswith("@"):
            low = line.lower()
            if "title" in low and "subtitle" not in low and "xaxis" not in low and "yaxis" not in low:
                val = _extract_quoted(line)
                if val:
                    title = val
            elif "subtitle" in low:
                val = _extract_quoted(line)
                if val:
                    subtitle = val
            elif "xaxis" in low and "label" in low:
                val = _extract_quoted(line)
                if val:
                    x_label = val
            elif "yaxis" in low and "label" in low:
                val = _extract_quoted(line)
                if val:
                    y_label = val
            else:
                m_leg = _LEGEND_RE.match(line)
                if m_leg:
                    legends[int(m_leg.group(1))] = m_leg.group(2)
                    continue
                m_type = _TYPE_RE.match(line)
                if m_type:
                    xy_type = m_type.group(1).lower()
            continue
        if line.startswith("&"):
            if current_block:
                blocks.append(current_block)
                current_block = []
            continue

        # Numeric data row (also tolerates trailing residue label e.g. gmx rama: -62.1 -41.5 ALA-12)
        parts = line.split()
        row: list[float] = []
        for p in parts:
            try:
                row.append(float(p))
            except ValueError:
                break
        if len(row) >= 2:
            current_block.append(row)

    if current_block:
        blocks.append(current_block)

    title_clean = grace_to_matplotlib(title)
    subtitle_clean = grace_to_matplotlib(subtitle)
    x_label_clean = grace_to_matplotlib(x_label) or "X"
    y_label_clean = grace_to_matplotlib(y_label) or "Y"

    stem = Path(filename).stem
    series_list: list[SeriesData] = []
    series_counter = 0

    for block_idx, block in enumerate(blocks):
        # Ensure rectangular rows (pad/truncate rare ragged lines to modal column count)
        col_counts = [len(r) for r in block]
        ncols = max(set(col_counts), key=col_counts.count)
        filtered = [r for r in block if len(r) == ncols]
        if not filtered:
            continue
        arr = np.asarray(filtered, dtype=float)
        x = arr[:, 0]

        # Interpret columns according to @TYPE (or -nxy)
        if xy_type == "xydy" and ncols == 3 and not nxy_hint:
            name = grace_to_matplotlib(legends.get(series_counter, "")) or (
                f"{stem}" if len(blocks) == 1 else f"{stem} #{block_idx + 1}"
            )
            series_list.append(
                SeriesData(
                    name=name,
                    x=x,
                    y=arr[:, 1],
                    dy=arr[:, 2],
                    source_file=filename,
                    series_index=series_counter,
                )
            )
            series_counter += 1
        elif xy_type == "xydx" and ncols == 3 and not nxy_hint:
            name = grace_to_matplotlib(legends.get(series_counter, "")) or f"{stem}"
            series_list.append(
                SeriesData(
                    name=name,
                    x=x,
                    y=arr[:, 1],
                    dx=arr[:, 2],
                    source_file=filename,
                    series_index=series_counter,
                )
            )
            series_counter += 1
        elif xy_type == "xydxdy" and ncols == 4 and not nxy_hint:
            name = grace_to_matplotlib(legends.get(series_counter, "")) or f"{stem}"
            series_list.append(
                SeriesData(
                    name=name,
                    x=x,
                    y=arr[:, 1],
                    dx=arr[:, 2],
                    dy=arr[:, 3],
                    source_file=filename,
                    series_index=series_counter,
                )
            )
            series_counter += 1
        else:
            # Multi-column xy / nxy (each column j >= 1 is a separate series)
            for col in range(1, ncols):
                raw_leg = legends.get(series_counter, "")
                if raw_leg:
                    name = grace_to_matplotlib(raw_leg)
                elif ncols == 2 and len(blocks) == 1:
                    name = stem
                else:
                    name = f"{stem} (s{series_counter})"
                series_list.append(
                    SeriesData(
                        name=name,
                        x=x,
                        y=arr[:, col],
                        source_file=filename,
                        series_index=series_counter,
                    )
                )
                series_counter += 1

    detected = detect_gromacs_type(filename, title, x_label, y_label, gmx_command)
    return ParsedDataset(
        filename=filename,
        file_format="xvg",
        kind="xy",
        title=title_clean,
        subtitle=subtitle_clean,
        x_label=x_label_clean,
        y_label=y_label_clean,
        x_unit=detect_unit(x_label) or detect_unit(x_label_clean),
        y_unit=detect_unit(y_label) or detect_unit(y_label_clean),
        detected_type=detected,
        gmx_command=gmx_command,
        series=series_list,
        metadata={
            "xy_type": xy_type,
            "num_blocks": len(blocks),
            "raw_title": title,
            "raw_x_label": x_label,
            "raw_y_label": y_label,
        },
    )


def parse_xvg(source: str | Path | TextIO, nxy_hint: bool = False) -> ParsedDataset:
    """Parse a .xvg file from path or file-like object."""
    if hasattr(source, "read"):
        name = getattr(source, "name", "data.xvg")
        text = source.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        return parse_xvg_text(text, filename=Path(str(name)).name, nxy_hint=nxy_hint)
    path = Path(source)
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_xvg_text(text, filename=path.name, nxy_hint=nxy_hint)
