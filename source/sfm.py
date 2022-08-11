from baseline import Baseline
import numpy as np
import cv2
import utils

class SFM:
    def __init__(self, views, matches):
        self.views = views
        self.matches = matches
        self.done=[]
        self.point_map = {}
        self.point_counter = 0
        self.points_3D = np.empty((0, 3))
        self.errors = []
        self.names=[]

        for view in self.views:
            self.names.append(view.name)

    
    def get_index_of_view(self, view):
        return self.names.index(view.name)


    def compute_pose(self, view1, view2=None, isBaseLine=False):
        if isBaseLine and view2:
            matchObject = self.matches[(view1.name, view2.name)]
            baselinePose = Baseline(view1, view2, matchObject)
            view2.R, view2.t = baselinePose.get_pose()
            rpe1, rpe2 = self.triangulate(view1, view2)
            self.errors.append(np.mean(rpe1))
            self.errors.append(np.mean(rpe2))

            self.done.append(view1)
            self.done.append(view2)
        else:
            view1.R, view1.t = self.compute_pose_PNP(view1)

    def reconstruct (self):
        baselineView1 = self.views[0]
        baselineView2 = self.views[1]
        self.computePose (view1=baselineView1, view2=baselineView2, isBaseLine=True)

    def triangulate (self, view1, view2):

        """Triangulates 3D points from two views whose poses have been recovered. Also updates the point_map dictionary"""

        K1_inv = np.linalg.inv(view1.K)
        K2_inv = np.linalg.inv(view2.K)
        P1 = np.hstack((view1.R, view1.t))
        P2 = np.hstack((view2.R, view2.t))

        match_object = self.matches[(view1.name, view2.name)]
        matchObject = self.matches[(view1.name, view2.name)]
        pixel_points1, pixel_points2 = matchObject.indices1, matchObject.indices2
        pixel_points1 = cv2.convertPointsToHomogeneous(pixel_points1)[:, 0, :]
        pixel_points2 = cv2.convertPointsToHomogeneous(pixel_points2)[:, 0, :]
        reprojection_error1 = []
        reprojection_error2 = []

        for i in range(len(pixel_points1)):

            u1 = pixel_points1[i, :]
            u2 = pixel_points2[i, :]

            u1_normalized = K1_inv.dot(u1)
            u2_normalized = K2_inv.dot(u2)

            point_3D = utils.get_3D_point(u1_normalized, P1, u2_normalized, P2)
            self.points_3D = np.concatenate((self.points_3D, point_3D.T), axis=0)

            error1 = utils.calculate_reprojection_error(point_3D, u1[0:2], self.K, view1.R, view1.t)
            reprojection_error1.append(error1)
            error2 = utils.calculate_reprojection_error(point_3D, u2[0:2], self.K, view2.R, view2.t)
            reprojection_error2.append(error2)

            # updates point_map with the key (index of view, index of point in the view) and value point_counter
            # multiple keys can have the same value because a 3D point is reconstructed using 2 points
            self.point_map[(self.get_index_of_view(view1), match_object.inliers1[i])] = self.point_counter
            self.point_map[(self.get_index_of_view(view2), match_object.inliers2[i])] = self.point_counter
            self.point_counter += 1

        return reprojection_error1, reprojection_error2

    def compute_pose_PNP(self, view):
        