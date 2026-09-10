"""Independent geometry and registration checks for the alignment correction."""
import unittest
import cv2
import numpy as np
from v11_realtime_tube import TubeGeometry
from v12_aligned_tracking import BackgroundStabilizer, RayTubeRefiner, VisibleRobotFilter, transform_uv


class AlignmentTests(unittest.TestCase):
    def test_ray_lift_reprojects_exactly_on_different_parts_of_anatomy(self):
        g=TubeGeometry()
        for i in [40,120,200,350,450,600,700]:
            uv=g.P[i]+np.array([1.,-2.])
            result=RayTubeRefiner(g).update(uv,g.s[i])
            self.assertEqual(result['status'],'within_radius_proxy')
            np.testing.assert_allclose(g.project(result['X']),uv,atol=1e-8)

    def test_incompatible_ray_is_flagged_and_stops_at_radius(self):
        g=TubeGeometry()
        result=RayTubeRefiner(g).update(g.P[450]+[180.,-180.],g.s[450])
        self.assertEqual(result['status'],'geometry_conflict')
        centre=np.array([np.interp(result['s'],g.s,g.C[:,j]) for j in range(3)])
        self.assertLessEqual(np.linalg.norm(result['X']-centre),result['radius']+1e-8)

    def test_background_registration_inverts_known_motion(self):
        cv2.setRNGSeed(12)
        rng=np.random.default_rng(12)
        frame=np.repeat(rng.integers(40,210,(240,320,1),dtype=np.uint8),3,axis=2)
        frame=cv2.GaussianBlur(frame,(3,3),.5)
        known=cv2.getRotationMatrix2D((160,120),.12,1.)
        known[:,2]+=[2.,-1.5]
        current=cv2.warpAffine(frame,known,(320,240),borderMode=cv2.BORDER_REFLECT_101)
        stabilizer=BackgroundStabilizer(frame,np.array([[15.,15.],[16.,16.]]))
        inverse,_,valid=stabilizer.update(current)
        self.assertTrue(valid)
        p=np.array([[80.,80.],[200.,160.],[160.,120.]])
        np.testing.assert_allclose(transform_uv(transform_uv(p,known),inverse),p,atol=.2)

    def test_missing_detection_is_not_reported_as_measured(self):
        f=VisibleRobotFilter()
        f.update(np.array([100.,100.]))
        uv,valid=f.update(np.array([np.nan,np.nan]))
        self.assertFalse(valid)
        self.assertTrue(np.isnan(uv).all())
        uv,valid=f.update(np.array([105.,100.]))
        self.assertTrue(valid)
        np.testing.assert_allclose(uv,[105.,100.])


if __name__=='__main__': unittest.main()
