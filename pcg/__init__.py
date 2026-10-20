"""pcg: code for the CirCor quantization trust-audit project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = PROJECT_ROOT / "splits"
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
