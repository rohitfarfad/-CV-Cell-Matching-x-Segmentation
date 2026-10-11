"""Contract tests for learned geometry targets, tiled inference and structural matching."""
import unittest
import numpy as np
import torch,cv2
from scipy import ndimage as ndi
from cell_models import CellUNet,loss_function,distance_target,dense_prediction,make_instances
from cell_alignment import assign,overlap_statistics,deform_points


class AdvancedTests(unittest.TestCase):
    def test_input_variants_preserve_native_shape_and_source_pixels(self):
        from unet_inputs import input_prediction
        from unet_cells import device
        model=CellUNet().to(device)
        for parameter in model.parameters():torch.nn.init.zeros_(parameter)
        source=np.arange(43*69,dtype=np.uint16).reshape(43,69)
        original=source.copy()
        for method in ['gamma_0.8','gamma_1.2','scale_0.85','scale_1.15','reflection','multiscale']:
            prediction=input_prediction(source,model,method)
            self.assertEqual(prediction.shape,(2,*source.shape))
            np.testing.assert_allclose(prediction,.5,atol=1e-6)
            np.testing.assert_array_equal(source,original)

    def test_fast_instance_rle_preserves_endpoints_and_cross_row_runs(self):
        from cell_local_matching import fast_instances,check_instances
        from cell_matching import instances
        labels=np.zeros((7,11),np.int32)
        labels[0,-2:]=1;labels[1,:3]=1
        labels[0,0]=2;labels[-1,-1]=2
        self.assertEqual(fast_instances(labels,'IV'),instances(labels,'IV'))
        self.assertEqual(fast_instances(labels,'IV')['IV_000001'],'9 5')
        self.assertEqual(fast_instances(np.zeros_like(labels),'IV'),{})
        check_instances(fast_instances(labels,'IV'),labels)
        with self.assertRaises(AssertionError):check_instances({'IV_000001':'76 2'},labels)
        with self.assertRaises(AssertionError):check_instances({'IV_000001':'9 2 11 3'},labels)

    def test_local_correction_reduces_smooth_drift_and_bounds_motion(self):
        from cell_local_matching import local_warp
        x,y=np.meshgrid(np.arange(20,180,20),np.arange(20,180,20))
        source=np.c_[x.ravel(),y.ravel()].astype(float)
        matrix=np.array([[1.,0.,10.],[0.,1.,15.]])
        affine=source+[10,15]
        target=affine+np.c_[2*np.sin(source[:,1]/40),2*np.cos(source[:,0]/45)]
        corrected=local_warp(source,target,matrix)
        self.assertLess(np.linalg.norm(corrected-target,axis=1).mean(),np.linalg.norm(affine-target,axis=1).mean())
        self.assertLessEqual(np.linalg.norm(corrected-affine,axis=1).max(),3+1e-9)
        np.testing.assert_array_equal(local_warp(source[:3],target[:3],matrix),affine[:3])

    def test_pair_calibration_keeps_unknown_correspondences_unlabeled(self):
        from cell_pair_scores import correspondence_labels,fit_calibration,pair_scores
        maps=({1:'a',2:'b',3:'c',5:'d'},{1:'x',2:'z',3:'w',4:'q',5:'y'})
        labels=correspondence_labels([(i,i) for i in range(1,6)],maps,{('a','x'),('b','y')})
        np.testing.assert_array_equal(labels,[1,0,-1,0,0])
        features=np.zeros((7,11));features[:,0]=[0,1,2,8,9,10,1000]
        calibration=fit_calibration(features,np.array([1,1,1,0,0,0,-1]))
        self.assertAlmostEqual(calibration['mean'][0],5)
        scores=pair_scores(features,calibration)
        self.assertTrue((scores[:3]>.5).all())
        self.assertTrue((scores[3:6]<.5).all())

    def test_quick_matching_rejects_forced_second_neighbor(self):
        from unet_quick import quick_pairs
        a=np.zeros((30,40),np.int32);b=np.zeros_like(a)
        a[10,10]=1;a[10,14]=2;b[10,11]=1;b[10,20]=2
        matrix=np.eye(2,3);statistics=dict(confidence=.5)
        self.assertEqual(quick_pairs([a,b],matrix,statistics),[(1,1),(2,2)])
        self.assertEqual(quick_pairs([a,b],matrix,statistics,mutual=True),[(1,1)])
        self.assertEqual(quick_pairs([a,b],matrix,dict(confidence=.1),mutual=True),[])
        self.assertEqual(quick_pairs([a,b],None,statistics),[])

    def test_distance_target_per_instance(self):
        labels=np.zeros((30,40),np.int32);labels[3:12,3:12]=1;labels[14:27,20:33]=2
        target=distance_target(labels)
        self.assertEqual(target[labels==1].max(),1)
        self.assertEqual(target[labels==2].max(),1)
        self.assertTrue((target[labels==0]==0).all())

    def test_heads_gradients_and_ray_guard(self):
        for kind,channels in [('distance',2),('stardist',33)]:
            model=CellUNet(kind);logits=model(torch.randn(1,1,31,37))
            target=torch.zeros_like(logits);loss=loss_function(logits,target,kind)
            loss.backward();self.assertEqual(logits.shape,(1,channels,31,37))
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaises(AssertionError):dense_prediction(np.zeros((31,37),np.uint16),CellUNet('stardist'),tile=32,augment=True)

    def test_constant_tta_and_weighted_tiles(self):
        from unet_cells import device
        model=CellUNet().to(device)
        for parameter in model.parameters():torch.nn.init.zeros_(parameter)
        for shape in [(1,1),(43,69)]:
            prediction=dense_prediction(np.zeros(shape,np.uint16),model,tile=32,augment=True)
            np.testing.assert_allclose(prediction,.5,atol=1e-6)
            self.assertEqual(prediction.shape,(2,*shape))

    def test_stardist_empty_and_single_polygon(self):
        prediction=np.zeros((33,40,50),np.float32)
        self.assertEqual(make_instances(prediction,'stardist',[5,200]).max(),0)
        prediction[0,20,25]=.9;prediction[1:,20,25]=5
        labels=make_instances(prediction,'stardist',[5,200])
        self.assertEqual(labels.max(),1);self.assertEqual(labels[20,25],1)

    def test_partial_overlap_and_structural_assignment(self):
        a=np.zeros((150,150),np.int32);b=np.zeros_like(a)
        points=np.array([[20,20],[50,20],[80,20],[20,50],[50,50],[80,50],[20,80],[50,80],[80,80]])
        for i,(x,y) in enumerate(points,1):
            cv2.circle(a,(int(x),int(y)),3,i,-1);cv2.circle(b,(int(x+10),int(y+15)),3,i,-1)
        # An unrelated target cell outside the overlap must not reduce confidence.
        cv2.circle(b,(140,140),3,10,-1)
        matrix=np.array([[1.,0.,10.],[0.,1.,15.]])
        confidence,support,close=overlap_statistics(points, np.vstack([points+[10,15],[140,140]]),matrix)
        self.assertAlmostEqual(confidence,1.);self.assertEqual(close,9)
        pairs=assign([a,b],points+[10,15],matrix,gate=6,structure=1)
        self.assertEqual(pairs,[(i,i) for i in range(1,10)])
        deformed=deform_points(points,np.vstack([points+[10,15],[140,140]]),matrix)
        np.testing.assert_allclose(deformed,points+[10,15],atol=1e-3)

    def test_cpd_reduces_smooth_local_drift(self):
        x,y=np.meshgrid(np.arange(20,180,20),np.arange(20,180,20))
        source=np.c_[x.ravel(),y.ravel()].astype(float)
        matrix=np.array([[1.,0.,10.],[0.,1.,15.]])
        affine=source+[10,15]
        target=affine+np.c_[2*np.sin(source[:,1]/40),2*np.cos(source[:,0]/45)]
        result=deform_points(np.vstack([source,[240.,240.]]),target,matrix)
        self.assertEqual(result.shape,(len(source)+1,2))
        self.assertTrue(np.isfinite(result).all())
        result=result[:-1]
        self.assertLess(np.linalg.norm(result-target,axis=1).mean(),np.linalg.norm(affine-target,axis=1).mean())


if __name__=='__main__':unittest.main()
