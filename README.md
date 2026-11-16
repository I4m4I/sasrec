# SASRec core implementation excerpt

This anonymous supplement presents selected logic from the study's SASRec
baseline implementation for inspection during review. It is a code excerpt;
dataset loading, experiment orchestration and reporting are outside its scope.

[`sasrec_core.py`](sasrec_core.py) contains background construction, the sequence
encoder, a training step, candidate scoring and validation checkpoint selection.
File-specific code is replaced by function arguments, the minibatch update is
wrapped as a function, and the checkpoint snapshot is kept in memory.

The background builder retains other users' interactions within the existing
item catalog. All validation and test users are first removed from the raw
background, then their exact saved history prefixes are restored. Their held-out
target events are excluded; the same item may legitimately occur in another
user's history. This follows a per-user holdout protocol without a global time
cutoff. Upstream processing verifies the item mapping and saved prefixes.

Training uses chronologically ordered sequences, left padding, a maximum input
length of 20, item dimension 64, two Transformer layers, two attention heads,
feedforward dimension 256, dropout 0.2, and sampled-softmax cross-entropy with
128 uniform negative draws. The caller supplies an Adam optimizer with learning
rate 0.001. Padding positions do not contribute training labels. The optimized
MHA fastpath is disabled, and non-finite losses or candidate scores raise an
error. Checkpoints are selected only by validation accuracy, retaining the
earliest epoch on ties. Test evaluation follows checkpoint selection.

The excerpt requires Python with PyTorch and pandas. It includes no datasets,
checkpoints, logs, predictions or experimental results.
