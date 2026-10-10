import unittest
import cv2
import numpy as np
from classical_cells import segment_cells, refine_boundaries, filter_regions, mutual_information, match_labels


class ClassicalRefinementTests(unittest.TestCase):
    def test_both_matching_families_allow_empty_regions(self):
        image=np.zeros((40,50),np.uint16);labels=np.zeros(image.shape,np.int32)
        for matcher in ['legacy','improved']:
            pairs,matrix,stats=match_labels([image,image],[labels,labels],
                dict(matcher=matcher,gate=8,cutoff=.2,structure=0))
            self.assertEqual(pairs,[]);self.assertIsNone(matrix)
            self.assertEqual(stats['confidence'],0)

    def test_empty_images_and_small_shapes(self):
        for method in ['baseline','local','otsu','contour','tophat','prominence','fiber','local_fiber']:
            for shape in [(1,1),(31,43)]:
                mask=segment_cells(np.zeros(shape,np.uint16),'exvivo',method)
                self.assertEqual(mask.shape,shape)
                self.assertEqual(mask.dtype,np.int32)
                self.assertEqual(mask.max(),0)

    def test_local_refinement_preserves_instance_separation(self):
        image=np.zeros((60,80),np.uint16);labels=np.zeros(image.shape,np.int32)
        for i,x in enumerate([27,40],1):
            cv2.circle(image,(x,30),6,50000,-1)
            cv2.circle(labels,(x,30),5,i,-1)
        for method in ['local','otsu','contour']:
            result=refine_boundaries(image,labels,method)
            self.assertEqual(result.max(),2)
            self.assertNotEqual(result[30,27],result[30,40])
            self.assertTrue(result[30,27]>0 and result[30,40]>0)

    def test_fiber_filter_keeps_bright_round_cell(self):
        image=np.zeros((70,80),np.uint16);labels=np.zeros(image.shape,np.int32)
        cv2.circle(image,(20,20),5,50000,-1);cv2.circle(labels,(20,20),5,1,-1)
        labels[45:50,45:65]=2  # Flat background candidate should be rejected.
        result=filter_regions(image,labels)
        self.assertEqual(result.max(),1);self.assertEqual(result[20,20],1)
        self.assertEqual(result[47,50],0)

    def test_mutual_information_handles_no_signal_and_transform_direction(self):
        image=np.zeros((100,120),np.uint16);matrix=np.array([[1.,0.,8.],[0.,1.,-3.]])
        result=mutual_information([image,image],matrix)
        np.testing.assert_allclose(result,matrix)
        rng=np.random.default_rng(4)
        source=cv2.GaussianBlur(rng.integers(1000,60000,(200,220),dtype=np.uint16),(0,0),2)
        target=cv2.warpAffine(source,matrix,(260,240))
        result=mutual_information([source,target],matrix)
        np.testing.assert_allclose(result,matrix,atol=1e-6)


if __name__=='__main__':unittest.main()
