"""Explicit input contracts, split checks and sequence tensor construction."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import SPLITS


@dataclass(frozen=True)
class Query:
    query_id: str
    user_id: str
    history: tuple[int, ...]
    target: int
    candidates: tuple[int, ...]


def integer_ids(values):
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Item IDs must be positive JSON integers; zero is padding")
    return tuple(values)


def load_catalog(directory):
    catalog = pd.read_csv(Path(directory) / "catalog.csv", dtype={"raw_item_id": str}, keep_default_na=False)
    if set(catalog.columns) != {"raw_item_id", "item_id"} or catalog.empty:
        raise ValueError("catalog.csv requires raw_item_id and item_id columns")
    if not pd.api.types.is_integer_dtype(catalog.item_id) or (catalog.raw_item_id == "").any():
        raise ValueError("Invalid catalog identifiers")
    if catalog.raw_item_id.duplicated().any() or catalog.item_id.duplicated().any():
        raise ValueError("Catalog mapping must be one-to-one")
    if sorted(catalog.item_id.tolist()) != list(range(1, len(catalog) + 1)):
        raise ValueError("Catalog item indices must be contiguous from 1")
    return dict(zip(catalog.raw_item_id, catalog.item_id))


def load_queries(directory, n_items, candidate_count):
    splits, used_users, used_keys = {}, set(), set()
    for split in SPLITS:
        queries = []
        with (Path(directory) / (split + ".jsonl")).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                history = integer_ids(row["history"])
                candidates = integer_ids(row["candidates"])
                target = integer_ids([row["target"]])[0]
                uid, key = row["user_id"], row["query_id"]
                if not isinstance(uid, str) or not uid or not isinstance(key, str) or not key:
                    raise ValueError("Query and user IDs must be nonempty strings")
                if uid in used_users or key in used_keys:
                    raise ValueError("Users and query IDs must be unique across all splits")
                if not history or len(candidates) != candidate_count or len(set(candidates)) != candidate_count:
                    raise ValueError("Each query requires a history and distinct fixed candidates")
                if target not in candidates or max(history + candidates + (target,)) > n_items:
                    raise ValueError("Target/candidate/catalog mismatch")
                used_users.add(uid)
                used_keys.add(key)
                queries.append(Query(key, uid, history, target, candidates))
        if not queries:
            raise ValueError("Empty evaluation split: " + split)
        splits[split] = queries
    return splits


def saved_prefixes(directory, splits, n_items):
    """Check saved queries against chronological histories, then drop targets."""
    frame = pd.read_csv(
        Path(directory) / "evaluation_histories.csv", dtype={"user_id": str}, keep_default_na=False
    )
    if set(frame.columns) != {"user_id", "item_id", "timestamp"}:
        raise ValueError("evaluation_histories.csv requires user_id, item_id, timestamp")
    if not pd.api.types.is_integer_dtype(frame.item_id) or not frame.item_id.between(1, n_items).all():
        raise ValueError("History item IDs must belong to the frozen catalog")
    frame["timestamp"] = pd.to_numeric(frame.timestamp, errors="raise")
    if not np.isfinite(frame.timestamp).all():
        raise ValueError("History timestamps must be finite")
    queries = {q.user_id: q for rows in splits.values() for q in rows}
    if set(frame.user_id) != set(queries):
        raise ValueError("Evaluation histories must contain exactly the selected evaluation users")
    frame = frame.sort_values("timestamp", kind="stable")
    prefixes = []
    for uid, events in frame.groupby("user_id", sort=True):
        q = queries[uid]
        if events.item_id.tolist() != list(q.history) + [q.target]:
            raise ValueError("Saved history/target does not match chronological events")
        prefixes.append(events.iloc[:-1].copy())
    return pd.concat(prefixes, ignore_index=True)


def sequence_matrix(background, max_length):
    sequences = []
    for _, events in background.groupby("user_id", sort=True):
        if len(events) >= 2:
            sequences.append(events.item_id.to_numpy(dtype=np.int64)[-(max_length + 1):])
    if not sequences:
        raise ValueError("No background users have at least two training interactions")
    matrix = np.zeros((len(sequences), max_length + 1), dtype=np.int64)
    for row, sequence in enumerate(sequences):
        matrix[row, -len(sequence):] = sequence
    return matrix


def training_tensors(sequences, device):
    sequences = torch.as_tensor(sequences, dtype=torch.long, device=device)
    x = sequences[:, :-1].contiguous()
    # A leading padding position must not predict the first observed item.
    y = torch.where(x > 0, sequences[:, 1:], 0).contiguous()
    if not torch.equal((y > 0).sum(1), (sequences > 0).sum(1) - 1):
        raise ValueError("Invalid next-item shift or sequence with fewer than two items")
    if not bool(((y > 0).sum(1) > 0).all()):
        raise ValueError("Every training sequence needs at least one next-item label")
    return x, y


def query_tensors(queries, max_length, device):
    x = torch.zeros((len(queries), max_length), dtype=torch.long, device=device)
    for row, query in enumerate(queries):
        history = query.history[-max_length:]
        x[row, -len(history):] = torch.tensor(history, device=device)
    candidates = torch.tensor([q.candidates for q in queries], dtype=torch.long, device=device)
    targets = torch.tensor([q.target for q in queries], dtype=torch.long, device=device)
    return x, candidates, targets
