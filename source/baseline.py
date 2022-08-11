import utils
import cv2
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
        K1_inv = np.linalg.inv(K1)
        K2_inv = np.linalg.inv(K2)
        P1 = np.hstack((self.view1.R, self.view1.t))
        P2 = np.hstack((R, t))

        # only reconstructs the inlier points filtered using the fundamental matrix
        pixel_points1, pixel_points2 = self.match_object.indices1, self.match_object.indices2

        # convert 2D pixel points to homogeneous coordinates
        pixel_points1 = cv2.convertPointsToHomogeneous(pixel_points1)[:, 0, :]
        pixel_points2 = cv2.convertPointsToHomogeneous(pixel_points2)[:, 0, :]

        reprojection_error = []

        points_3D = np.zeros((0, 3))  # stores the triangulated points

        for i in range(len(pixel_points1)):
            u1 = pixel_points1[i, :]
            u2 = pixel_points2[i, :]

            # convert homogeneous 2D points to normalized device coordinates
            u1_normalized = K1_inv.dot(u1)
            u2_normalized = K2_inv.dot(u2)

            # calculate 3D point
            point_3D = utils.get_3D_point(u1_normalized, P1, u2_normalized, P2)

            # calculate reprojection error
            error = utils.calculate_reprojection_error(point_3D, u2[0:2], K, R, t)
            reprojection_error.append(error)

            # append point
            points_3D = np.concatenate((points_3D, point_3D.T), axis=0)

        return np.mean(reprojection_error), points_3D