"""Core implementation excerpt: background construction, training and scoring.

Paths, dataset readers, experiment orchestration and reporting are omitted.
Item indices start at 1; 0 is reserved for left padding.
"""
import pandas as pd
import torch
from torch import nn


def build_training_background(raw_chunks, item_to_idx, saved_prefixes):
    """Restore exact evaluation-user prefixes after excluding their raw rows.

    raw_chunks: frames with user_id, parent_asin and timestamp.
    item_to_idx: the existing catalog's item-to-index mapping.
    saved_prefixes: user_id, item_id and timestamp for all validation/test
    users' saved histories; user IDs use the raw namespace. Held-out target
    events are already excluded from these prefixes.
    """
    evaluation_users = set(saved_prefixes.user_id)
    columns = ["user_id", "item_id", "timestamp"]
    parts = []
    for chunk in raw_chunks:
        keep = chunk.parent_asin.isin(item_to_idx) & ~chunk.user_id.isin(evaluation_users)
        subset = chunk.loc[keep].copy()
        subset["item_id"] = subset.parent_asin.map(item_to_idx).astype("int32")
        parts.append(subset[columns])
    background = pd.concat(parts + [saved_prefixes[columns]], ignore_index=True)
    return background.sort_values("timestamp", kind="stable")


class SASRec(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        # Use the non-fastpath implementation for left-padded causal inputs.
        torch.backends.mha.set_fastpath_enabled(False)
        self.emb = nn.Embedding(vocabulary_size, 64, padding_idx=0)
        self.pos = nn.Embedding(20, 64)
        layer = nn.TransformerEncoderLayer(64, 2, 256, 0.2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.ln = nn.LayerNorm(64)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        hidden = self.emb(x) + self.pos(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            x.size(1), device=x.device
        )
        return self.ln(self.enc(hidden, mask=causal_mask, src_key_padding_mask=(x == 0)))


def train_step(model, optimizer, padded_sequences):
    """One batch of next-item training; each sequence has at least two items."""
    model.train()
    sequences = padded_sequences[:, -21:]
    x = sequences[:, :-1].contiguous()
    y = torch.where(x > 0, sequences[:, 1:], 0).contiguous()
    hidden = model(x)
    valid = y > 0
    representations, targets = hidden[valid], y[valid]
    # Preserve the original independent, uniform training-negative draws.
    negatives = torch.randint(
        1, model.emb.num_embeddings, (len(targets), 128), device=x.device
    )
    positive_scores = (representations * model.emb(targets)).sum(-1, keepdim=True)
    negative_scores = torch.bmm(
        model.emb(negatives), representations.unsqueeze(-1)
    ).squeeze(-1)
    logits = torch.cat((positive_scores, negative_scores), dim=1)
    loss = nn.functional.cross_entropy(
        logits, torch.zeros(len(targets), dtype=torch.long, device=x.device)
    )
    if not torch.isfinite(loss).all():
        raise FloatingPointError("Non-finite training loss")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


@torch.no_grad()
def rank_candidates(model, histories, candidates):
    """Rank fixed candidates for nonempty, left-padded histories of length 20."""
    model.eval()
    torch.backends.mha.set_fastpath_enabled(False)
    representation = model(histories)[:, -1]
    scores = (model.emb(candidates) * representation[:, None, :]).sum(-1)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Non-finite candidate scores; refusing to rank")
    return candidates.gather(1, scores.argmax(-1, keepdim=True)).squeeze(1)


def keep_validation_checkpoint(model, validation_correct, best_correct, best_state):
    """Called after each epoch; initialize best_correct=-1 and best_state=None.

    Strict improvement keeps the earliest checkpoint when validation ties.
    Test labels are not inputs to checkpoint selection.
    """
    if validation_correct > best_correct:
        best_correct = validation_correct
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_correct, best_state
