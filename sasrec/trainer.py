"""Background training followed by evaluation of a validation-selected model."""
import json
import random
from pathlib import Path

import numpy as np
import torch

from .checkpoint import ValidationSelection, load_checkpoint, save_checkpoint
from .config import input_fingerprints, require_files, sha256_file, write_json
from .data import load_catalog, load_queries, training_tensors
from .evaluation import evaluate
from .model import SASRec
from .objectives import next_item_loss


def select_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.mha.set_fastpath_enabled(False)


def verify_inputs(data_dir, manifest):
    require_files(data_dir)
    if input_fingerprints(data_dir) != manifest["input_sha256"]:
        raise ValueError("Input files changed after background preparation")


def evaluate_checkpoint(checkpoint_path, data_dir, device):
    state = load_checkpoint(checkpoint_path, device)
    verify_inputs(data_dir, state["manifest"])
    cfg = state["config"]
    splits = load_queries(data_dir, state["n_items"], cfg["candidate_count"])
    model = SASRec(state["n_items"], **cfg["model"]).to(device)
    model.load_state_dict(state["model"])
    queries = splits["test"] + splits["test2"]
    return evaluate(model, queries, cfg["model"]["max_length"], device)


def train(cfg, data_dir, output_dir, device):
    output_dir = Path(output_dir)
    require_files(output_dir, ("background.npz", "manifest.json"))
    manifest = json.loads((output_dir / "manifest.json").read_text())
    verify_inputs(data_dir, manifest)
    if any(manifest[key] != cfg[key] for key in ("domain", "background")):
        raise ValueError("Prepared background does not match the requested domain/protocol")
    if manifest["max_length"] != cfg["model"]["max_length"]:
        raise ValueError("Prepared sequence length does not match the model")
    if sha256_file(output_dir / "background.npz") != manifest["background_sha256"]:
        raise ValueError("Prepared background checksum mismatch")
    catalog = load_catalog(data_dir)
    if len(catalog) != manifest["n_items"]:
        raise ValueError("Catalog size differs from the prepared background")
    splits = load_queries(data_dir, len(catalog), cfg["candidate_count"])
    with np.load(output_dir / "background.npz", allow_pickle=False) as archive:
        x_all, y_all = training_tensors(archive["sequences"], device)
    run_dir = output_dir / "fit"
    run_dir.mkdir(exist_ok=False)
    settings = cfg["training"]
    set_seed(settings["seed"])
    model = SASRec(len(catalog), **cfg["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    selection = ValidationSelection()
    trajectory = []
    checkpoint = run_dir / "best.pt"
    write_json(run_dir / "config.json", cfg)
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(x_all), device=device)
        total_loss = torch.zeros((), device=device)
        total_labels = 0
        for start in range(0, len(order), settings["batch_size"]):
            indices = order[start:start + settings["batch_size"]]
            loss, labels = next_item_loss(model, x_all[indices], y_all[indices], settings["negative_samples"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * labels
            total_labels += labels
        validation = evaluate(model, splits["val"], cfg["model"]["max_length"], device)
        if selection.consider(validation.correct, epoch):
            save_checkpoint(checkpoint, model, cfg, len(catalog), manifest, epoch)
        trajectory.append({"epoch": epoch, "loss": float(total_loss / total_labels),
                           "validation_correct": validation.correct, "validation_n": validation.total})
        write_json(run_dir / "training.json", trajectory)
        if selection.should_stop(epoch, settings["min_epochs"], settings["patience"]):
            break
    # Test scores never enter the epoch loop or checkpoint-selection decision.
    state = load_checkpoint(checkpoint, device)
    model.load_state_dict(state["model"])
    test = evaluate(model, splits["test"] + splits["test2"], cfg["model"]["max_length"], device)
    write_json(run_dir / "evaluation.json", {
        "selected_epoch": state["epoch"], "test_correct": test.correct,
        "test_n": test.total, "test_accuracy": test.accuracy,
    })
