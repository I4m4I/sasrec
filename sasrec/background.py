"""Recover background interactions while preserving all held-out query prefixes."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import input_fingerprints, require_files, sha256_file, write_json
from .data import load_catalog, load_queries, saved_prefixes, sequence_matrix


def merge_background(chunks, item_to_idx, prefixes):
    evaluation_users = set(prefixes.user_id)
    columns = ["user_id", "item_id", "timestamp"]
    parts = []
    for chunk in chunks:
        keep = chunk.raw_item_id.isin(item_to_idx) & ~chunk.user_id.isin(evaluation_users)
        selected = chunk.loc[keep].copy()
        selected["item_id"] = selected.raw_item_id.map(item_to_idx).astype("int64")
        selected["timestamp"] = pd.to_numeric(selected.timestamp, errors="raise")
        if not np.isfinite(selected.timestamp).all() or (selected.user_id == "").any():
            raise ValueError("Invalid background timestamp or user ID")
        parts.append(selected[columns])
    # Restore only the exact saved prefixes, never the evaluation target events.
    background = pd.concat(parts + [prefixes[columns]], ignore_index=True)
    if background.user_id.isin(evaluation_users).sum() != len(prefixes):
        raise ValueError("Evaluation-user background contamination")
    return background.sort_values("timestamp", kind="stable")


def prepare(cfg, data_dir, output_dir):
    require_files(data_dir)
    output_dir = Path(output_dir)
    if (output_dir / "background.npz").exists() or (output_dir / "manifest.json").exists():
        raise FileExistsError("Prepared background already exists; use a new output root")
    catalog = load_catalog(data_dir)
    splits = load_queries(data_dir, len(catalog), cfg["candidate_count"])
    prefixes = saved_prefixes(data_dir, splits, len(catalog))
    chunks = pd.read_csv(
        Path(data_dir) / "interactions.csv",
        usecols=["user_id", "raw_item_id", "timestamp"],
        dtype={"user_id": str, "raw_item_id": str},
        keep_default_na=False,
        chunksize=cfg["chunk_size"],
    )
    background = merge_background(chunks, catalog, prefixes)
    sequences = sequence_matrix(background, cfg["model"]["max_length"])
    degree = np.bincount(background.item_id.to_numpy(), minlength=len(catalog) + 1)
    manifest = {
        "domain": cfg["domain"], "background": cfg["background"],
        "max_length": cfg["model"]["max_length"], "n_items": len(catalog),
        "input_sha256": input_fingerprints(data_dir),
        "protocol": "per-user holdout; no global timestamp cutoff",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "background.npz"
    np.savez_compressed(artifact, sequences=sequences, degree=degree)
    manifest["background_sha256"] = sha256_file(artifact)
    write_json(output_dir / "manifest.json", manifest)
