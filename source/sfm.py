from ssl import DefaultVerifyPaths
from baseline import Baseline
import numpy as np
import cv2
import utils
import os

# match_technique = 'SIFT'
match_technique = 'LoFTR'

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
        
        if not os.path.exists(self.views[0].dataset_path + '/points'):
            os.makedirs(self.views[0].dataset_path + '/points')

        # store results in a root_path/points
        self.results_path = os.path.join(self.views[0].dataset_path, 'points')

    
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
            if match_technique == 'LoFTR':
                view1.R, view1.t = self.compute_pose_PNP(view1)
            elif match_technique == 'SIFT':
                view1.R, view1.t = self.compute_pose_PNP_SIFT(view1)
            errors=[]

            for i, old_view in enumerate(self.done):
                match_object = self.matches[(old_view.name, view1.name)]
                self.remove_mapped_points(match_object, i)
                _, rpe = self.triangulate(old_view, view1)
                errors += rpe
            self.done.append(view1)
            self.errors.append(np.mean(errors))

    def reconstruct (self):
        baselineView1 = self.views[0]
        baselineView2 = self.views[1]
        self.compute_pose(view1=baselineView1, view2=baselineView2, isBaseLine=True)
        self.plot_points()
        for i in range(2, len(self.views)):
            view = self.views[i]
            self.compute_pose(view1=self.views[i])
            if match_technique == 'SIFT':
                self.plot_points()
        

    def triangulate (self, view1, view2):

        """Triangulates 3D points from two views whose poses have been recovered. Also updates the point_map dictionary"""

        K1_inv = np.linalg.inv(view1.K)
        K2_inv = np.linalg.inv(view2.K)
        P1 = np.hstack((view1.R, view1.t))
        P2 = np.hstack((view2.R, view2.t))

        match_object = self.matches[(view1.name, view2.name)]
        pixel_points1, pixel_points2 = utils.get_pixel_points(match_object)   #matchOject
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

            error1 = utils.calculate_reprojection_error(point_3D, u1[0:2], view2.K, view1.R, view1.t)
            reprojection_error1.append(error1)
            error2 = utils.calculate_reprojection_error(point_3D, u2[0:2], view1.K, view2.R, view2.t)
            reprojection_error2.append(error2)

            # updates point_map with the key (index of view, index of point in the view) and value point_counter
            # multiple keys can have the same value because a 3D point is reconstructed using 2 points
            self.point_map[(self.get_index_of_view(view1), match_object.inliers1[i])] = self.point_counter
            self.point_map[(self.get_index_of_view(view2), match_object.inliers2[i])] = self.point_counter
            self.point_counter += 1

        return reprojection_error1, reprojection_error2

    def compute_pose_PNP(self, view):
        points_3D, points_2D = np.zeros((0, 3)), np.zeros((0, 2))
        m=[]
        for i, old_view in enumerate(self.done):
            match_object=self.matches[(old_view.name, view.name)]
            if i == 0:
                for mo in match_object.matches:
                    m.append([i,mo])
            else:
                pp2_list = [ p[1].pixel_points2 for p in m]
                for j in range(len(match_object.matches)):
                    idx, cond = self.checkPoints(match_object.matches[j].pixel_points2, pp2_list)
                    if cond:
                        if m[idx][1].confidence < match_object.matches[j].confidence:
                            m[idx][0] = i
                            m[idx][1] = match_object.matches[j]
                        


        match_sorted= sorted(m,key=lambda x:x[1].queryIdx)
        for match in match_sorted:
            old_image_idx, new_image_kp_idx, old_image_kp_idx = match[0], match[1].queryIdx, match[1].trainIdx
            if (old_image_idx,old_image_kp_idx) in self.point_map:
                
                try:
                    point_2D=match_object.pixel_points2[new_image_kp_idx].T.reshape(1,2)
                    points_2D=np.concatenate((points_2D,point_2D),axis=0) 
                    point_3D = self.points_3D[self.point_map[(old_image_idx, old_image_kp_idx)], :].T.reshape((1, 3))
                    points_3D = np.concatenate((points_3D, point_3D), axis=0)
                except:
                    pass

        _, R, t, _ = cv2.solvePnPRansac(points_3D[:, np.newaxis], points_2D[:, np.newaxis], view.K, None,
                                confidence=0.99, reprojectionError=8.0, flags=cv2.SOLVEPNP_DLS)
        R, _ = cv2.Rodrigues(R)

        return R, t

    def checkPoints(self, mp, checkList):
        for i in range(len(checkList)):
            if mp[0] == checkList[i][0] and mp[1] == checkList[i][1]:
                return i, True
        return -1, False



    def compute_pose_PNP_SIFT(self, view):
        points_3D, points_2D = np.zeros((0, 3)), np.zeros((0, 2))
        
        m=[]
        for i, old_view in enumerate(self.done):
            match_object=self.matches[(old_view.name, view.name)]
            if i == 0:
                for mo in match_object.matches:
                    m.append([i,mo])
            else:
                for j in range(len(match_object.matches)):
                    if m[j][1].distance > match_object.matches[j].distance:
                        m[j][0]=i
                        m[j][1]=match_object.matches[j]

       
        match_sorted= sorted(m,key=lambda x:x[1].queryIdx)
        for match in match_sorted:
            old_image_idx, new_image_kp_idx, old_image_kp_idx = match[0], match[1].queryIdx, match[1].trainIdx
            if (old_image_idx,old_image_kp_idx) in self.point_map:
                point_2D=match_object.pixel_points2[new_image_kp_idx].T.reshape(1,2)
                points_2D=np.concatenate((points_2D,point_2D),axis=0) 
                point_3D = self.points_3D[self.point_map[(old_image_idx, old_image_kp_idx)], :].T.reshape((1, 3))
                points_3D = np.concatenate((points_3D, point_3D), axis=0)
        _, R, t, _ = cv2.solvePnPRansac(points_3D[:, np.newaxis], points_2D[:, np.newaxis], view.K, None,
                                confidence=0.99, reprojectionError=8.0, flags=cv2.SOLVEPNP_DLS)
        R, _ = cv2.Rodrigues(R)

        return R, t





    def plot_points(self):
            """Saves the reconstructed 3D points to ply files using Open3D"""

            number = len(self.done)
            filename = os.path.join(self.results_path, str(number) + '_images.xyz')
            for point in self.points_3D:
                with open(filename, 'a') as f:
                    f.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')
                    f.close()
        
    def remove_mapped_points(self, match_object, image_idx):
        """Removes points that have already been reconstructed in the completed views"""

        inliers1 = []
        inliers2 = []

        for i in range(len(match_object.inliers1)):
            if (image_idx, match_object.inliers1[i]) not in self.point_map:
                inliers1.append(match_object.inliers1[i])
                inliers2.append(match_object.inliers2[i])

        match_object.inliers1 = inliers1
        match_object.inliers2 = inliers2