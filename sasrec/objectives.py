"""Historical sampled-softmax objective with uniform training negatives."""
import torch
from torch.nn import functional as F

from .model import require_finite


def next_item_loss(model, x, y, negative_samples):
    hidden = model(x)
    valid = y > 0
    representation, target = hidden[valid], y[valid]
    if target.numel() == 0:
        raise ValueError("Training minibatch contains no valid next-item labels")
    # Independent draws with replacement, matching the historical objective.
    # No target/history rejection is added to the training sampler.
    negatives = torch.randint(
        1, model.emb.num_embeddings, (len(target), negative_samples), device=x.device
    )
    positive = (representation * model.emb(target)).sum(-1, keepdim=True)
    negative = torch.bmm(model.emb(negatives), representation.unsqueeze(-1)).squeeze(-1)
    logits = torch.cat((positive, negative), dim=1)
    loss = F.cross_entropy(logits, torch.zeros(len(target), dtype=torch.long, device=x.device))
    require_finite(loss, "training loss")
    return loss, len(target)
