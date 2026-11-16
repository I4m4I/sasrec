"""Configuration loading and data preflight, without importing ML libraries."""
import copy
import hashlib
import json
from pathlib import Path

SPLITS = ("val", "test", "test2")
INPUT_NAMES = ("catalog.csv", "interactions.csv", "evaluation_histories.csv") + tuple(
    split + ".jsonl" for split in SPLITS
)


def load_config(path, domain):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if domain not in raw["domains"]:
        raise ValueError("Unknown domain; use one of: " + ", ".join(raw["domains"]))
    cfg = {key: copy.deepcopy(raw[key]) for key in ("model", "training", "candidate_count", "chunk_size")}
    cfg["domain"] = domain
    cfg["background"] = raw["domains"][domain]["background"]
    cfg["training"].update(raw["domains"][domain].get("training", {}))
    m, t = cfg["model"], cfg["training"]
    for key in ("max_length", "dimension", "layers", "heads", "feedforward"):
        if not isinstance(m[key], int) or m[key] < 1:
            raise ValueError("Model setting must be a positive integer: " + key)
    if m["dimension"] % m["heads"] or not 0 <= m["dropout"] < 1:
        raise ValueError("Invalid attention dimension or dropout")
    for key in ("batch_size", "negative_samples", "max_epochs", "min_epochs", "patience"):
        if not isinstance(t[key], int) or t[key] < 1:
            raise ValueError("Training setting must be a positive integer: " + key)
    if t["min_epochs"] > t["max_epochs"] or t["learning_rate"] <= 0:
        raise ValueError("Invalid epoch limits or learning rate")
    if cfg["candidate_count"] < 2 or cfg["chunk_size"] < 1:
        raise ValueError("Invalid candidate count or input chunk size")
    return cfg


def require_files(directory, names=INPUT_NAMES):
    missing = [name for name in names if not (Path(directory) / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Experiment inputs are not distributed with this repository. Missing in "
            + str(directory) + ": " + ", ".join(missing)
            + ". See docs/data_contract.md; no automatic download or synthetic fallback is used."
        )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprints(directory):
    return {name: sha256_file(Path(directory) / name) for name in INPUT_NAMES}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
