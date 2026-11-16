# Implementation and evaluation protocol

## Background recovery

`background.merge_background` filters the raw interaction stream to the original
item catalog. It excludes all selected validation and test users before combining
the remaining stream with their exact saved prefixes. `data.saved_prefixes`
verifies those prefixes against the saved chronological sequences and removes
the final target event.

Excluding a target event does not mean excluding that item's entire identity.
The same item can occur earlier in the user's prefix or in another user's
history. Such legitimate interactions remain available to the model. Catalog
membership alone does not establish that an item had positive training examples.

This is the historical per-user holdout protocol. It has no global timestamp
cutoff across users and must not be described as a globally chronological
new-product benchmark. The expanded-background procedure changes data coverage;
Books uses its original background with a longer training schedule.

## Sequence model and padding

The network is the study's Transformer-based SASRec baseline implementation.
Item and positional embeddings feed two causal Transformer encoder layers,
followed by layer normalization. This is a retained study implementation, not a
claim that the source is the original authors' reference implementation.

Sequences are left padded. The input contains the most recent configured number
of items; training labels are the one-position shift of that sequence. The
`torch.where(x > 0, shifted_items, 0)` construction prevents padding from creating
a label for the first observed item. Users with fewer than two training events
do not contribute next-item training sequences.

The optimized multi-head attention evaluation path is disabled explicitly.
In the investigated runtime, left padding together with a causal mask caused
non-finite scores on that path. The same encoder, embeddings and masks are
retained, and the non-fastpath implementation is used. This is a runtime-specific
correction, not a claim that every PyTorch version has the same behavior.

Training rejects non-finite loss values. Evaluation checks both the final
representation and candidate scores before `argmax`. It raises an error instead
of substituting a candidate when numerical scoring is invalid.

## Training objective

For each valid next-item position, the objective compares its positive score
against independent uniform item draws using cross-entropy. The positive is
placed in the first column of the logits. The sampler retains the historical
behavior: draws are with replacement, exclude padding, and do not reject a draw
merely because it matches the positive item or a history item.

The optimizer is Adam. Each epoch shuffles the background-user sequence rows.
The architecture and learning-rate defaults are in the configuration. The
training loop uses the latest sequence suffix, as in the study's background
training script; it does not expand every sequence into all possible windows.

## Validation and test separation

After an epoch, evaluate the saved validation queries on their unchanged
candidate sets. Save a checkpoint only on a strict increase in validation
correctness. Since the validation set is fixed, this is equivalent to selecting
by validation accuracy. Ties keep the earlier checkpoint. Early stopping uses
the configured minimum epoch count and patience.

Test correctness is never computed inside the training loop. After epoch
selection, restore the selected checkpoint and evaluate the combined held-out
test queries. The evaluation representation is the encoder output at the last
sequence position; candidate scores are embedding dot products. This preserves
the original candidate order, including its tie-breaking behavior.

## Refactoring boundaries

The module layout, argument parsing, input schema validation and input fingerprint
checks are additions for this source release. Absolute paths, private readers,
hardware-specific launch commands, historical logs and result tables are omitted.
The refactoring retains the original model and optimization operations.

The numerical correction can be examined independently. The later background
retraining also changed the training batch size, runtime/device and checkpoint
selection. It should not be presented as a controlled one-factor ablation of
background size. No performance claim or result table is included here.
