import utils
import numpy as np

class Baseline:

    def __init__(self, view1, view2, match_object):
        self.view1 = view1
        self.view1.R = np.eye(3,3)
        self.view2 = view2
        self.match_object = match_object
    
    def get_pose(self):
        R1,R2,t1,t2 = utils.get_extrinsic_from_E(self.math_object.E)

        if not utils.check_determinant(R1):
            R1,R2,t1,t2 = utils.get_extrinsic_from_E(-self.math_object.E)

        reprojection_error, points_3D = self.triangulate(K1= self.view1.K,K2= self.view2.K, R= R1, t= t1)
        
    
    def triangulate(self, K1, K2, R, t):
        K_inv = np.linalg.inv(K)
        points_3D = utils.triangulate_points(self.match_object.points1, self.match_object.points2, K, R, t)
        reprojection_error = utils.get_reprojection_error(points_3D, self.match_object.points1, self.match_object.points2, K)
        return reprojection_error, points_3D
