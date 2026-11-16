"""Implementation checks using only generated in-memory tensors and frames."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sasrec.background import merge_background
from sasrec.checkpoint import ValidationSelection, load_checkpoint, save_checkpoint
from sasrec.config import require_files
from sasrec.data import Query, sequence_matrix, training_tensors
from sasrec.evaluation import candidate_scores, evaluate
from sasrec.model import SASRec, require_finite
from sasrec.objectives import next_item_loss


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(0)

    def test_background_restores_only_saved_prefix(self):
        raw = pd.DataFrame({
            "user_id": ["heldout", "heldout", "background", "background"],
            "raw_item_id": ["first", "target", "target", "outside"],
            "timestamp": [1, 2, 3, 4],
        })
        prefix = pd.DataFrame({"user_id": ["heldout"], "item_id": [1], "timestamp": [1]})
        result = merge_background([raw], {"first": 1, "target": 2}, prefix)
        self.assertEqual(result[result.user_id == "heldout"].item_id.tolist(), [1])
        self.assertEqual(result[result.user_id == "background"].item_id.tolist(), [2])

    def test_shift_does_not_predict_first_item_from_padding(self):
        sequences = np.zeros((2, 21), dtype=np.int64)
        sequences[0, -3:] = [1, 2, 3]
        sequences[1] = np.arange(1, 22)
        x, y = training_tensors(sequences, "cpu")
        self.assertEqual((y > 0).sum(1).tolist(), [2, 20])
        self.assertTrue(bool((y[x == 0] == 0).all()))

    def test_sequence_suffix_and_short_user_filter(self):
        frame = pd.DataFrame({"user_id": ["long"] * 7 + ["short"], "item_id": list(range(1, 8)) + [1]})
        matrix = sequence_matrix(frame, max_length=3)
        self.assertEqual(matrix.tolist(), [[4, 5, 6, 7]])

    def test_short_and_full_histories_have_finite_scores(self):
        model = SASRec(40)
        x = torch.zeros(2, 20, dtype=torch.long)
        x[0, -3:] = torch.arange(1, 4)
        x[1] = torch.arange(1, 21)
        candidates = torch.arange(1, 11).expand(2, -1)
        torch.backends.mha.set_fastpath_enabled(True)
        scores = candidate_scores(model, x, candidates)
        self.assertTrue(bool(torch.isfinite(scores).all()))
        self.assertFalse(torch.backends.mha.get_fastpath_enabled())

    def test_invalid_scores_fail_before_ranking(self):
        with self.assertRaises(FloatingPointError):
            require_finite(torch.tensor([float("nan")]), "test")
        model = SASRec(40)
        with torch.no_grad():
            model.emb.weight[1].fill_(float("nan"))
        query = Query("check", "user", (2, 3), 1, tuple(range(1, 11)))
        with self.assertRaises(FloatingPointError):
            evaluate(model, [query], 20, "cpu")

    def test_training_loss_and_gradients_are_finite(self):
        model = SASRec(40)
        sequences = np.zeros((1, 21), dtype=np.int64)
        sequences[0, -4:] = [1, 2, 3, 4]
        x, y = training_tensors(sequences, "cpu")
        loss, labels = next_item_loss(model, x, y, negative_samples=128)
        self.assertEqual(labels, 3)
        loss.backward()
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))

    def test_validation_ties_keep_first_epoch(self):
        selection = ValidationSelection()
        self.assertTrue(selection.consider(1, 1))
        self.assertFalse(selection.consider(1, 2))
        self.assertEqual(selection.best_epoch, 1)
        self.assertFalse(selection.should_stop(2, 3, 1))
        self.assertTrue(selection.should_stop(3, 3, 1))
        self.assertTrue(selection.consider(2, 4))
        self.assertEqual(selection.best_epoch, 4)

    def test_checkpoint_round_trip(self):
        model = SASRec(40)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            save_checkpoint(path, model, {}, 40, {}, 1)
            state = load_checkpoint(path, "cpu")
            self.assertTrue(all(torch.equal(value, state["model"][key]) for key, value in model.state_dict().items()))

    def test_missing_data_has_no_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "not distributed"):
                require_files(temporary)
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
