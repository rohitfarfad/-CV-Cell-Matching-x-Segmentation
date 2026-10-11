"""Neural inference contracts; accuracy is measured separately by mouse."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from unet_cells import SmallUNet, loss_function, probabilities, instances_from_probabilities


class UNetTests(unittest.TestCase):
    def test_refinement_preserves_default_instance_contract(self):
        from unet_refined import refined_instances
        probability=np.zeros((2,30,40),np.float32)
        probability[0,6:22,4:30]=.9
        probability[1,9:19,7:14]=.9
        probability[1,9:19,20:27]=.9
        np.testing.assert_array_equal(refined_instances(probability,[5,500]),
                                      instances_from_probabilities(probability,[5,500]))
        for method in ['threshold','smooth','peaks','probability']:
            empty=refined_instances(np.zeros_like(probability),[5,500],method=method)
            self.assertEqual(empty.max(),0)

    def test_smoothing_rejects_isolated_confident_pixel(self):
        from unet_refined import refined_instances
        probability=np.zeros((2,25,30),np.float32)
        probability[:,5:12,5:12]=.9
        probability[:,20,24]=.9
        raw=refined_instances(probability,[1,500],threshold=.6)
        smooth=refined_instances(probability,[1,500],threshold=.6,method='smooth')
        self.assertGreater(raw[20,24],0)
        self.assertEqual(smooth[20,24],0)
        self.assertGreater(smooth[8,8],0)

    def test_gradients_and_checkpoint(self):
        torch.manual_seed(7)
        model = SmallUNet()
        image = torch.randn(2, 1, 31, 37)
        result = model(image)
        self.assertEqual(result.shape, (2, 2, 31, 37))
        loss_function(result, torch.zeros_like(result)).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pt'
            torch.save(model.state_dict(), path)
            restored = SmallUNet()
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(model(image), restored(image))

    def test_tiling_coverage(self):
        # A constant-output probe makes uncovered pixels and overlap errors observable.
        class Constant(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((len(x), 2, *x.shape[-2:]), device=x.device)
        for shape in [(1, 1), (25, 39), (69, 83)]:
            result = probabilities(np.zeros(shape, np.uint16), Constant(), tile=32, stride=24)
            self.assertEqual(result.shape, (2, *shape))
            np.testing.assert_array_equal(result, .5)

    def test_touching_instances_and_empty(self):
        probability = np.zeros((2, 30, 40), np.float32)
        probability[0, 6:22, 4:30] = .9
        probability[1, 9:19, 7:14] = .9
        probability[1, 9:19, 20:27] = .9
        labels = instances_from_probabilities(probability, [5, 500], seeds='components')
        self.assertEqual(labels.max(), 2)
        np.testing.assert_array_equal(labels > 0, probability[0] > .5)
        empty = instances_from_probabilities(np.zeros_like(probability), [5, 500])
        self.assertEqual(empty.max(), 0)


if __name__ == '__main__':
    unittest.main()
