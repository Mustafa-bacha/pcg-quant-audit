"""Zip the code needed on Kaggle/Colab into dist/pcg_code.zip (no data, no venv, no runs).

    .venv\\Scripts\\python.exe scripts\\make_code_bundle.py

Upload the zip to Kaggle as a Dataset (or to Colab's file panel) and run
notebooks/kaggle_p1_train.ipynb.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ["pcg", "scripts", "splits", "third_party", "requirements.txt", "README.md"]
OUT = ROOT / "dist" / "pcg_code.zip"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            p = ROOT / item
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
            for f in files:
                z.write(f, f.relative_to(ROOT).as_posix())
                n += 1
    print(f"wrote {OUT} ({n} files, {OUT.stat().st_size / 1024:.0f} KB)")
    sys.exit(0)


if __name__ == "__main__":
    main()
