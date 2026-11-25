"""Build static GitHub Pages bundle for GMXgrace Studio (Pyodide WebAssembly mode)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC_SRC = ROOT / "src" / "gmxgrace" / "web" / "static"
OUT_DIR = ROOT / "dist_pages"


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    # 1. Copy static Web Studio UI files
    for item in STATIC_SRC.iterdir():
        dest = OUT_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    # 2. Build pure-Python wheel directly into dist_pages/wheels/
    wheels_dir = OUT_DIR / "wheels"
    wheels_dir.mkdir(parents=True)
    if shutil.which("uv"):
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(wheels_dir), str(ROOT)],
            check=True,
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheels_dir), str(ROOT)],
            check=True,
        )

    wheels = sorted(wheels_dir.glob("gmxgrace-*.whl"))
    if not wheels:
        raise RuntimeError("Failed to build gmxgrace wheel for GitHub Pages!")
    wheel_name = wheels[0].name

    # 3. Write wheel_manifest.json and .nojekyll
    manifest = {
        "wheel": f"./wheels/{wheel_name}",
        "engine": "pyodide",
    }
    (OUT_DIR / "wheel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / ".nojekyll").write_text("", encoding="utf-8")

    print(f"Built GitHub Pages bundle in {OUT_DIR}: wheel={wheel_name}")


if __name__ == "__main__":
    main()
