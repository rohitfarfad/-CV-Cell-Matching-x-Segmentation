"""Run in the isolated Cellpose environment; no pretrained download in these tests."""
import unittest
import numpy as np
import torch
from cellpose_cells import boundary_dice_loss,instances_from_flows


class CellposeContracts(unittest.TestCase):
    def test_boundary_dice_finite_gradients(self):
        for foreground in [False,True]:
            target=np.zeros((2,3,31,37),np.float32)
            if foreground:target[:,0,8:22,10:25]=1
            prediction=torch.zeros(2,3,31,37,requires_grad=True)
            loss=boundary_dice_loss(target,prediction,torch.device('cpu'))
            loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_empty_flows(self):
        mask=instances_from_flows((np.zeros((2,25,37),np.float32),np.full((25,37),-100,np.float32)),{'diameter':10.,'limits':[5,300]})
        self.assertEqual(mask.shape,(25,37));self.assertEqual(mask.max(),0)
        self.assertEqual(mask.dtype,np.int32)


if __name__=='__main__':unittest.main()
