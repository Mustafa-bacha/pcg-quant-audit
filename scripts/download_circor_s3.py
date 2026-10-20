"""Download CirCor v1.0.3 from the public AWS Open Data mirror (physionet-open bucket).

Used because physionet.org itself served ~4 KB/s from the dev network on 2026-10-20.
Resumable: files already present with the right size are skipped.

    .venv\\Scripts\\python.exe scripts\\download_circor_s3.py
"""

from __future__ import annotations

import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcg import RAW_DIR  # noqa: E402

BUCKET = "https://physionet-open.s3.amazonaws.com"
# Any PhysioNet open dataset works: pass the bucket prefix and a destination folder name, e.g.
#   python scripts/download_circor_s3.py --prefix challenge-2016/1.0.0/ --dest physionet-2016
PREFIX = "circor-heart-sound/1.0.3/"
DEST = RAW_DIR / "circor-heart-sound-1.0.3"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
WORKERS = 24
if "--prefix" in sys.argv:
    PREFIX = sys.argv[sys.argv.index("--prefix") + 1]
if "--dest" in sys.argv:
    DEST = RAW_DIR / sys.argv[sys.argv.index("--dest") + 1]
# --only a,b : keep only keys whose path (after the prefix) starts with one of these
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None


def list_keys() -> list[tuple[str, int]]:
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": PREFIX, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for c in root.findall("s3:Contents", NS):
            keys.append((c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text)))
        if root.find("s3:IsTruncated", NS).text != "true":
            break
        token = root.find("s3:NextContinuationToken", NS).text
    return keys


def fetch(session: requests.Session, key: str, size: int) -> int:
    out = DEST / key[len(PREFIX):]
    if out.exists() and out.stat().st_size == size:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            r = session.get(f"{BUCKET}/{key}", timeout=120)
            r.raise_for_status()
            if len(r.content) != size:
                raise IOError(f"size mismatch {len(r.content)} != {size}")
            out.write_bytes(r.content)
            return size
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    return 0


def main() -> None:
    keys = list_keys()
    if ONLY:
        keys = [(k, s) for k, s in keys if any(k[len(PREFIX):].startswith(o) for o in ONLY)]
    total = sum(s for _, s in keys)
    print(f"{len(keys)} objects, {total / 1e6:.0f} MB under s3://physionet-open/{PREFIX}", flush=True)
    done_bytes, n_done, t0 = 0, 0, time.time()
    with requests.Session() as s, ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fetch, s, k, sz): k for k, sz in keys}
        for f in as_completed(futs):
            done_bytes += f.result()
            n_done += 1
            if n_done % 500 == 0 or n_done == len(keys):
                el = time.time() - t0
                print(f"{n_done}/{len(keys)} files, {done_bytes / 1e6:.0f} MB new, "
                      f"{done_bytes / 1e6 / max(el, 1e-9):.2f} MB/s, {el:.0f}s", flush=True)
    missing = [(k, sz) for k, sz in keys
               if not (DEST / k[len(PREFIX):]).exists() or (DEST / k[len(PREFIX):]).stat().st_size != sz]
    print(f"complete: {len(keys) - len(missing)}/{len(keys)} files verified by size; missing={len(missing)}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
