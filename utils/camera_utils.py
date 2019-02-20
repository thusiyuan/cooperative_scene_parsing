"""
Created on 1 April, 2017

@author: Siyuan Huang

scripts about camera transformation
"""

"""
Camera notes:
    K: the intrinsic camera parameter
    R_ex: the standard extrinsic camera parameter
    R_tilt: the extrinsic camera parameter after changing the coordinate from left-hand to right-hand
    T:  translation
    
    [u, v, 1].T = K [R_ex T] [x_w, y_w, z_w, 1].T
    Rtilt = [1 0 0; 0 0 1 ;0 -1 0]*R_ex*[1 0 0; 0 0 -1 ;0 1 0];
    
    p_world.dot(R_ex.T) = p_camera
    R_ex.T.dot(p_camera)[0, 2, -1] = p_world_right
"""
import numpy as np
from numpy.linalg import inv
import copy
from shapely.geometry.polygon import Polygon


def rotation_matrix_3d_z(angle):
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(angle)
    R[1, 1] = np.cos(angle)
    R[0, 1] = -np.sin(angle)
    R[1, 0] = np.sin(angle)
    R[2, 2] = 1
    return R


def rotation_matrix_3d_x(angle):
    R = np.zeros((3, 3))
    R[1, 1] = np.cos(angle)
    R[2, 2] = np.cos(angle)
    R[1, 2] = -np.sin(angle)
    R[2, 1] = np.sin(angle)
    R[0, 0] = 1
    return R


def rotation_matrix_3d_y(angle):
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(angle)
    R[1, 1] = 1
    R[0, 2] = np.sin(angle)
    R[2, 0] = -np.sin(angle)
    R[2, 2] = np.cos(angle)
    return R


def calibrate_cam(vp, h, w):
    """
    Estimate the camera parameters
    Check dhoiem.cs.illinois.edu/courses/vision_spring10/lectures/lecture3_projectivegeomtery.pdf for detail algorithm

    Parameters
    ----------
    vp : numpy array
        vanishing points
    h: float
        height of the image
    w: float
        width of the image
    Return
    ---------
    k: 3x3 numpy array
        intrinsic camera parameter
    r: 3x3 numpy array
        extrinsic camera parameter
    """
    infchk = np.logical_and(vp[:, 0] > 50*w, vp[:, 1] > 50*h)
    if np.sum(infchk) == 0:
        v1 = vp[0, :]
        v2 = vp[1, :]
        v3 = vp[2, :]
        m_11 = v1[0] + v2[0]
        m_12 = v1[1] + v2[1]
        m_13 = v1[0]*v2[0] + v1[1]*v2[1]
        m_21 = v1[0] + v3[0]
        m_22 = v1[1] + v3[1]
        m_23 = v1[0]*v3[0] + v1[1]*v3[1]
        m_31 = v3[0] + v2[0]
        m_32 = v3[1] + v2[1]
        m_33 = v3[0]*v2[0] + v3[1]*v2[1]
        a_11 = m_11 - m_21
        a_12 = m_12 - m_22
        a_21 = m_11 - m_31
        a_22 = m_12 - m_32
        b_1 = m_13 - m_23
        b_2 = m_13 - m_33
        det_a = a_11*a_22 - a_12*a_21
        u0 = (a_22*b_1-a_12*b_2)/det_a
        v0 = (a_11*b_2-a_21*b_1)/det_a
        temp = m_11*u0 + m_12*v0 - m_13 - u0*u0 - v0*v0
        f = temp ** 0.5
    if np.sum(infchk) == 1:
        ii = np.nonzero(infchk == 0)
        v1 = vp[ii[0], :]
        v2 = vp[ii[1], :]
        r = ((w/2 - v1[0])*(v2[0] - v1[0])+(h/2 - v1[1])*(v2[1]-v1[1])) / ((v2[0] - v1[0]) ** 2 + (v2[1] - v1[1]) ** 2)
        u0 = v1[0] + r*(v2[0] - v1[0])
        v0 = v1[1] + r*(v2[1] - v1[1])
        temp = u0 * (v1[0] + v2[0]) + v0*(v2[1] + v1[1]) - (v1[0]*v2[0] + v2[1]*v1[1] + u0**2 + v0**2)
        f = temp ** 0.5
    if 'f' in locals() and 'u0' in locals() and 'v0' in locals():
        k = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]])
        vecx = inv(k).dot(np.hstack((vp[1, :], 1)).T)
        vecx /= np.linalg.norm(vecx)
        if vp[1, 0] < u0:
            vecx = -vecx
        vecz = inv(k).dot(np.hstack((vp[2, :], 1)).T)
        vecz = -vecz/np.linalg.norm(vecz)
        vecy = np.cross(vecz, vecx)
        r = np.vstack((vecx, vecy, vecz))
    if sum(infchk) == 2:
        if infchk[1] == 1:
            vp[1, :] = vp[1, :]/np.linalg.norm(vp[1, :])
        if infchk[0] == 1:
            vp[0, :] = vp[0, :]/np.linalg.norm(vp[0, :])
        if infchk[2] == 1:
            vp[2, :] = vp[2, :]/np.linalg.norm(vp[2, :])
        u0 = w/2
        v0 = h/2
        if infchk[1] == 1:
            vecx = np.hstack((vp[1, :], 0)).T
            vecx /= np.linalg.norm(vecx)
            if vp[1, 0] < u0:
                vecx = -vecx
        if infchk[0] == 1:
            vecy = np.hstack((vp[0, :], 0)).T
            vecy /= np.linalg.norm(vecy)
            if vp[0, 1] > v0:
                vecy = -vecy
        if infchk[2] == 1:
            vecz = np.hstack((vp[2, :], 0)).T
            vecz = -vecz/np.linalg.norm(vecz)
        if 'vecx' in locals() and 'vecy' in locals():
            vecz = np.cross(vecx, vecy)
        elif 'vecy' in locals() and 'vecz' in locals():
            vecx = np.cross(vecy, vecz)
        else:
            vecy = np.cross(vecz, vecx)
        r = np.vstack((vecx, vecy, vecz))
        f = 544
        k = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]])
    return k, r


def single_image_to_world(C, p, K, R, h):
    """
    Compute the right world coordinate location from the rgb coordinates given height
    [X, Y, Z]^T = C + \lambda R^-1 K^-1 p

    Parameters
    ----------
    C : np.array([c_x, c_y, c_z])
        3d center of the camera
    p: np.array([u, v, 1])
        2d location in the image
    K: 3x3 numpy array
        intrinsic camera parameter
    R: 3x3 numpy array
        extrinsic camera parameter
    h: float
        height of the 3d points
    Return
    ---------
    loc_3d: np.array([x, y, z])
        location of the 3d point in right-hand coordinate
    """
    temp_foot = inv(R).dot(inv(K)).dot(p)
    temp_foot = temp_foot[[0, 2, 1]]
    temp_foot[2] *= -1
    scale = (h - C[2]) / temp_foot[2]
    x_foot = scale * temp_foot[0]
    y_foot = scale * temp_foot[1]
    z_foot = h
    loc_3d = np.array([x_foot, y_foot, z_foot])
    return loc_3d


def rgbd_to_world(p, depth, K, R_tilt, R=None):
    """
    Compute the right world coordinate location from the pixel position and depth

    Parameters
    ----------
    p: np.array([x, y])
        2d location in image
    K: 3x3 numpy array
        intrinsic camera parameter
    R_tilt: 3x3 numpy array
        extrinsic camera parameter in right-hand coordinates
    R: 3x3 numpy array
        extrinsic camera parameter in left-hand coordinates
    Return
    ---------
    new_coor: np.array([x, y, z])
        location of the 3d point in right-hand coordinate
    """
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    x = p[0]
    y = p[1]
    x3 = (x + 1 - cx) * depth / fx
    y3 = (y + 1 - cy) * depth / fy
    z3 = depth
    new_coor = R_tilt.T.dot(np.array([x3, z3, -y3]))   # from camera coordinate to world coordinate so it's R_tilt.T
    # R.T.dot(np.array([x3, y3, z3]))[0, 2, -1]   # this is equal to the last sentence
    return new_coor


def normalize_point(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def yaw_pitch_row_from_r(r):
    """
    get the angle of yaw, pitch and row from rotation matrix
    http://planning.cs.uiuc.edu/node103.html
    :param r: numpy array
            rotation matrix
    :return:yaw, pitch, roll: float
            angle of yaw, pitch and roll
    """
    yaw = np.arctan(r[1, 0] / r[0, 0])
    pitch = np.arctan(-r[2, 0] / np.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))
    roll = np.arctan(r[2, 1] / r[2, 2])
    return yaw, pitch, roll


def points_3d_to_2d(gt3dcorners, K, R_ex):
    """
    :param gt3dcorners: n x 3 numpy array
    :param K: intrinsic parameter
    :param R_ex: extrinsic rotation matrixs
    :return: gt2dcorners: n x 2 numpy arrays
    """
    gt_3dcorners = gt3dcorners.T
    gt_3dcorners_temp = R_ex.dot(gt_3dcorners)
    gt_3dcorners_temp[2, :] = np.abs(gt_3dcorners_temp[2, :])
    gt2dcorners = K.dot(gt_3dcorners_temp)  # note the gt3dcorner
    gt2dcorners = gt2dcorners[:2, :] / gt2dcorners[2, :]
    return gt2dcorners.T


def get_rotation_matrix_from_yaw_roll(yaw, roll):
    """
    :param yaw:
    :param roll:
    :return: R_ex in left hand coordinates
    """
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(yaw)
    R[0, 1] = - np.sin(yaw) * np.cos(roll)
    R[0, 2] = np.sin(roll) * np.sin(yaw)
    R[1, 0] = np.sin(yaw)
    R[1, 1] = np.cos(roll) * np.cos(yaw)
    R[1, 2] = - np.cos(yaw) * np.sin(roll)
    R[2, 0] = 0
    R[2, 1] = np.sin(roll)
    R[2, 2] = np.cos(roll)
    return R
