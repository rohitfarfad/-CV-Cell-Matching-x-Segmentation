"""Small contract tests; these do not measure real-image accuracy."""
import unittest
import json
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from cell_matching import decode_rle, encode_rle, mask_quality, summarize, register, match_cells, centers, evaluate


class CellMatchingTests(unittest.TestCase):
    def test_rle(self):
        for mask in [np.zeros((3, 4), bool), np.ones((3, 4), bool),
                     np.isin(np.arange(12), [0, 3, 4, 5, 11]).reshape(3, 4)]:
            np.testing.assert_array_equal(mask, decode_rle(encode_rle(mask), mask.shape))
        for invalid in ['-1 2', '10 3', '0 0', '3 2 4 2']:
            with self.assertRaises(AssertionError):
                decode_rle(invalid, (3, 4))

    def test_strict_iou_and_false_positives(self):
        pred = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
        truth = {'a': '0 3'}
        self.assertEqual(mask_quality(pred, truth), (0., {}))  # Exactly .75 is not a TP.
        pred[0, 3] = 0
        self.assertEqual(mask_quality(pred, truth), (1., {1: 'a'}))
        pred[1, 0] = 2
        self.assertAlmostEqual(mask_quality(pred, truth)[0], 2/3)
        self.assertEqual(mask_quality(pred, {})[0], 0.)

    def test_pair_metric(self):
        mask = np.array([[1, 1], [0, 0]], dtype=np.int32)
        row = pd.DataFrame([dict(invivo_instances=json.dumps({"a": "0 2"}),
                                 exvivo_instances=json.dumps({"b": "0 2"}),
                                 match_pairs=json.dumps([["a", "b"]]))]).itertuples(index=False)
        result = evaluate([mask, mask], [(1, 1)], next(row))
        self.assertEqual(result["correct_pairs"], 1)
        self.assertEqual(result["recoverable_pairs"], 1)

    def test_pooled_f1(self):
        frame = pd.DataFrame(dict(pq_invivo=[1., 1.], pq_exvivo=[1., 1.],
                                  correct_pairs=[1, 0], predicted_pairs=[1, 9], verified_pairs=[1, 1]))
        self.assertAlmostEqual(summarize(frame)['f1_match'], 1/6)

    def test_empty_and_unmatched(self):
        empty = np.zeros((30, 30), np.int32)
        self.assertEqual(centers(empty).shape, (0, 2))
        self.assertEqual(match_cells(empty, empty, None)[0], [])
        a, b = empty.copy(), empty.copy()
        for i, (y, x) in enumerate([(5, 5), (15, 5), (5, 15)], 1):
            a[y, x] = i
            b[y+2, x+3] = i
        b[25, 25] = 4
        pairs, _ = match_cells(a, b, np.array([[1., 0., 3.], [0., 1., 2.]]), gate=2)
        self.assertEqual(pairs, [(1, 1), (2, 2), (3, 3)])

    def test_registration_translation(self):
        rng = np.random.default_rng(7)
        a = np.zeros((128, 128), np.float32)
        xy = rng.integers(10, 118, (35, 2))
        a[xy[:, 0], xy[:, 1]] = rng.uniform(.5, 1, len(xy))
        a = ndi.gaussian_filter(a, 2)
        a = (a/a.max()*65535).astype(np.uint16)
        b = np.zeros((256, 256), np.uint16)
        b[60:188, 40:168] = a
        matrix, score = register(a, b)
        error = np.linalg.norm(xy[:, ::-1] @ matrix[:, :2].T + matrix[:, 2] - (xy[:, ::-1]+[40, 60]), axis=1)
        self.assertLess(np.median(error), 3)
        self.assertGreater(score, .5)


if __name__ == '__main__':
    unittest.main()
