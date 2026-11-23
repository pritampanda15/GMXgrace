"""Parser for binary GROMACS .edr energy files via panedr/pyedr."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from gmxgrace.models import ParsedDataset, SeriesData


def _load_edr_path(path: Path, columns: list[str] | None = None) -> ParsedDataset:
    try:
        import pyedr  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Parsing GROMACS binary .edr files requires 'panedr' / 'pyedr'. "
            "Install with: pip install 'gmxgrace[edr]'"
        ) from exc

    data = pyedr.edr_to_dict(str(path), verbose=False)
    units = pyedr.get_unit_dictionary(str(path)) if hasattr(pyedr, "get_unit_dictionary") else {}

    time_key = "Time" if "Time" in data else next(iter(data.keys()))
    x = np.asarray(data[time_key], dtype=float)
    available_cols = [k for k in data.keys() if k != time_key]

    if columns:
        selected = [c for c in columns if c in data]
        if not selected:
            # Try case-insensitive partial match
            lower_map = {k.lower(): k for k in available_cols}
            for req in columns:
                rlow = req.lower()
                if rlow in lower_map:
                    selected.append(lower_map[rlow])
                else:
                    matches = [k for k in available_cols if rlow in k.lower()]
                    selected.extend(matches[:1])
        if not selected:
            raise ValueError(f"Requested .edr columns {columns} not found. Available: {available_cols}")
    else:
        selected = available_cols

    series: list[SeriesData] = []
    for idx, col in enumerate(selected):
        unit = units.get(col, "")
        label = f"{col} ({unit})" if unit else col
        series.append(
            SeriesData(
                name=label,
                x=x,
                y=np.asarray(data[col], dtype=float),
                source_file=path.name,
                series_index=idx,
                metadata={"raw_term": col, "unit": unit},
            )
        )

    y_unit = units.get(selected[0], "") if len(selected) == 1 else None
    y_label = f"{selected[0]} ({y_unit})" if (len(selected) == 1 and y_unit) else "Energy / Observable"

    return ParsedDataset(
        filename=path.name,
        file_format="edr",
        kind="xy",
        title=f"GROMACS Energies ({path.stem})",
        x_label="Time (ps)",
        y_label=y_label,
        x_unit="ps",
        y_unit=y_unit or None,
        detected_type="energy",
        series=series,
        metadata={"available_terms": available_cols, "units": dict(units)},
    )


def parse_edr(path: str | Path, columns: list[str] | None = None) -> ParsedDataset:
    """Parse a binary .edr file from disk."""
    return _load_edr_path(Path(path), columns=columns)


def parse_edr_bytes(raw: bytes, filename: str = "ener.edr", columns: list[str] | None = None) -> ParsedDataset:
    """Parse uploaded binary .edr bytes via a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".edr", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        ds = _load_edr_path(Path(tmp.name), columns=columns)
        ds.filename = filename
        for s in ds.series:
            s.source_file = filename
        return ds
