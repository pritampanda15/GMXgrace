"""Parser for generic scientific tabular files (.csv, .tsv, .dat, .txt)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from gmxgrace.models import ParsedDataset, SeriesData
from gmxgrace.parsers.xvg import detect_gromacs_type, detect_unit


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def parse_tabular_text(text: str, filename: str = "data.dat") -> ParsedDataset:
    """Parse CSV, TSV, or whitespace-delimited .dat/.txt numeric tables."""
    ext = Path(filename).suffix.lower()
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith(("#", "%", "@", ";"))]
    if not raw_lines:
        raise ValueError(f"No data rows found in {filename}")

    # Detect delimiter
    if ext == ".csv":
        delimiter: str | None = ","
    elif ext == ".tsv":
        delimiter = "\t"
    else:
        sample = raw_lines[0]
        if "," in sample and len(sample.split(",")) >= 2:
            delimiter = ","
        elif "\t" in sample and len(sample.split("\t")) >= 2:
            delimiter = "\t"
        else:
            delimiter = None  # whitespace

    def _split(line: str) -> list[str]:
        if delimiter is None:
            return line.split()
        return next(csv.reader([line], delimiter=delimiter))

    first_tokens = [t.strip() for t in _split(raw_lines[0])]
    has_header = not all(_is_float(tok) for tok in first_tokens if tok)

    headers: list[str]
    data_lines: list[str]
    if has_header:
        headers = first_tokens
        data_lines = raw_lines[1:]
    else:
        headers = [f"Col {i}" for i in range(len(first_tokens))]
        data_lines = raw_lines

    rows: list[list[float]] = []
    for ln in data_lines:
        toks = [t.strip() for t in _split(ln)]
        try:
            vals = [float(t) for t in toks if t != ""]
        except ValueError:
            continue
        if len(vals) >= 2:
            rows.append(vals)

    if not rows:
        raise ValueError(f"No valid numeric rows found in {filename}")

    col_counts = [len(r) for r in rows]
    ncols = max(set(col_counts), key=col_counts.count)
    arr = np.asarray([r for r in rows if len(r) == ncols], dtype=float)

    while len(headers) < ncols:
        headers.append(f"Col {len(headers)}")

    x = arr[:, 0]
    x_label = headers[0] if has_header else "X"
    y_label = headers[1] if (has_header and ncols == 2) else "Value"
    stem = Path(filename).stem

    series: list[SeriesData] = []
    for col in range(1, ncols):
        name = headers[col] if has_header else (stem if ncols == 2 else f"{stem} (col {col})")
        series.append(
            SeriesData(
                name=name,
                x=x,
                y=arr[:, col],
                source_file=filename,
                series_index=col - 1,
            )
        )

    return ParsedDataset(
        filename=filename,
        file_format="tabular",
        kind="xy",
        title=stem,
        x_label=x_label,
        y_label=y_label,
        x_unit=detect_unit(x_label),
        y_unit=detect_unit(y_label),
        detected_type=detect_gromacs_type(filename, stem, x_label, y_label, ""),
        series=series,
    )


def parse_tabular(path: str | Path) -> ParsedDataset:
    """Parse a tabular file from disk."""
    p = Path(path)
    return parse_tabular_text(p.read_text(encoding="utf-8", errors="replace"), filename=p.name)
