"""Evaluate unchanged candidate sets; reject invalid scores before argmax."""
from dataclasses import dataclass

import torch

from .data import query_tensors
from .model import require_finite


@dataclass(frozen=True)
class Evaluation:
    correct: int
    total: int
    predictions: tuple[int, ...]

    @property
    def accuracy(self):
        return self.correct / self.total


@torch.no_grad()
def candidate_scores(model, histories, candidates):
    if histories.ndim != 2 or not bool((histories[:, -1] > 0).all()):
        raise ValueError("Evaluation histories must be nonempty and left padded")
    model.eval()
    torch.backends.mha.set_fastpath_enabled(False)
    representation = model(histories)[:, -1]
    require_finite(representation, "last-position representations")
    scores = (model.emb(candidates) * representation[:, None, :]).sum(-1)
    require_finite(scores, "candidate scores; refusing to rank")
    return scores


def evaluate(model, queries, max_length, device):
    if not queries:
        raise ValueError("Cannot evaluate an empty split")
    histories, candidates, targets = query_tensors(queries, max_length, device)
    scores = candidate_scores(model, histories, candidates)
    chosen = candidates.gather(1, scores.argmax(-1, keepdim=True)).squeeze(1)
    return Evaluation(int((chosen == targets).sum()), len(queries), tuple(chosen.cpu().tolist()))
