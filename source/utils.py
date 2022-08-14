import numpy as np
import cv2

def get_extrinsic_from_E (E):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    W_t = W.T
    u, w, vt = np.linalg.svd(E)
    R1= u @ W @ vt
    R2 = u @ W_t @ vt
    t1 = u[:, -1].reshape((3, 1))
    t2 = - t1
    return R1, R2, t1, t2

def check_determinant(R):
    if np.linalg.det(R) + 1.0 < 1e-9:
        return False
    else:
        return True

def get_3D_point(u1, P1, u2, P2):
    """Solves for 3D point using homogeneous 2D points and the respective camera matrices"""

    A = np.array([[u1[0] * P1[2, 0] - P1[0, 0], u1[0] * P1[2, 1] - P1[0, 1], u1[0] * P1[2, 2] - P1[0, 2]],
                  [u1[1] * P1[2, 0] - P1[1, 0], u1[1] * P1[2, 1] - P1[1, 1], u1[1] * P1[2, 2] - P1[1, 2]],
                  [u2[0] * P2[2, 0] - P2[0, 0], u2[0] * P2[2, 1] - P2[0, 1], u2[0] * P2[2, 2] - P2[0, 2]],
                  [u2[1] * P2[2, 0] - P2[1, 0], u2[1] * P2[2, 1] - P2[1, 1], u2[1] * P2[2, 2] - P2[1, 2]]])

    B = np.array([-(u1[0] * P1[2, 3] - P1[0, 3]),
                  -(u1[1] * P1[2, 3] - P1[1, 3]),
                  -(u2[0] * P2[2, 3] - P2[0, 3]),
                  -(u2[1] * P2[2, 3] - P2[1, 3])])

    X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
    return X[1]

def calculate_reprojection_error(point_3D, point_2D, K, R, t):
    """Calculates the reprojection error for a 3D point by projecting it back into the image plane"""

    reprojected_point = K.dot(R.dot(point_3D) + t)
    reprojected_point = cv2.convertPointsFromHomogeneous(reprojected_point.T)[:, 0, :].T
    error = np.linalg.norm(point_2D.reshape((2, 1)) - reprojected_point)
    return error

def check_triangulation(points, P):
    """Checks whether reconstructed points lie in front of the camera"""

    P = np.vstack((P, np.array([0, 0, 0, 1])))
    reprojected_points = cv2.perspectiveTransform(src=points[np.newaxis], m=P)
    z = reprojected_points[0, :, -1]
    if (np.sum(z > 0)/z.shape[0]) < 0.75:
        return False
    else:
        return True


def remove_outliers_using_F(view1, view2, match_object):
    """Removes outlier keypoints using the fundamental matrix"""

    pixel_points1, pixel_points2 = match_object.pixel_points1, match_object.pixel_points2
    F, mask = cv2.findFundamentalMat(pixel_points1, pixel_points2, method=cv2.FM_RANSAC,
                                     ransacReprojThreshold=0.9, confidence=0.99)
    mask = mask.astype(bool).flatten()
    match_object.inliers1 = np.array(match_object.pixel_points1)[mask]
    match_object.inliers2 = np.array(match_object.pixel_points2)[mask]

    return F

def get_pixel_points(match_object):
    pixel_points1=np.array(match_object.pixel_points1)[match_object.inliers1]
    pixel_points2=np.array(match_object.pixel_points2)[match_object.inliers2]
    return pixel_points1, pixel_points2