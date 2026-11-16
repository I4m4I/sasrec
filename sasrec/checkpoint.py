"""Validation-only epoch selection, with strict improvement and early stopping."""
from dataclasses import dataclass

import torch


@dataclass
class ValidationSelection:
    best_correct: int = -1
    best_epoch: int = 0

    def consider(self, correct, epoch):
        if correct > self.best_correct:
            self.best_correct, self.best_epoch = correct, epoch
            return True
        return False

    def should_stop(self, epoch, minimum_epochs, patience):
        return epoch >= minimum_epochs and epoch - self.best_epoch >= patience


def save_checkpoint(path, model, cfg, n_items, manifest, epoch):
    payload = {
        "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "config": cfg, "n_items": n_items, "manifest": manifest, "epoch": epoch,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)
