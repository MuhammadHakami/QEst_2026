"""Folder layout: everything lives inside QEst_2026/."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET = ROOT / 'dataset'          # the competition's files (read-only)
RAW = ROOT / 'data' / 'raw'         # fetched once: OSM, weather, Elering
DERIVED = ROOT / 'data' / 'derived'  # built tables (15-min frame, simulated production)
RESULTS = ROOT / 'results'
