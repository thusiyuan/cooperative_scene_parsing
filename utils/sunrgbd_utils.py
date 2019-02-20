"""
Created on 1 April, 2017

@author: Siyuan Huang

scripts used in SUNRGBD Dataset

"""
import numpy as np
from camera_utils import normalize_point
from shapely.geometry.polygon import Polygon


def flip_towards_viewer(normals, points):
    points = points / np.linalg.norm(points)
    proj = points.dot(normals[:2, :].T)
    flip = np.where(proj > 0)
    normals[flip, :] = -normals[flip, :]
    return normals


def get_corners_of_bb3d(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    # order the basis
    index = np.argsort(np.abs(basis[:, 0]))[::-1]
    # the case that two same value appear the same time
    if index[2] != 2:
        index[1:] = index[1:][::-1]
    basis = basis[index, :]
    coeffs = coeffs[index]
    # Now, we know the basis vectors are orders X, Y, Z. Next, flip the basis vectors towards the viewer
    basis = flip_towards_viewer(basis, centroid)
    coeffs = np.abs(coeffs)
    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners = corners + np.tile(centroid, (8, 1))
    return corners


def get_corners_of_bb3d_no_index(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    coeffs = np.abs(coeffs)
    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners = corners + np.tile(centroid, (8, 1))
    return corners


def get_bdb_from_corners(corners):
    """
    get coeffs, basis, centroid from corners
    :param corners: 8x3 numpy array
        corners of a 3D bounding box
    :return: bounding box parameters
    """
    z_max = np.max(corners[:, 2])
    z_min = np.min(corners[:, 2])
    points_2d = corners[corners[:, 2] == z_max, :2]
    points_2d = points_2d[np.argsort(points_2d[:, 0]), :]
    vector1 = np.array([points_2d[1, 0] - points_2d[0, 0], points_2d[1, 1] - points_2d[0, 1], 0])
    vector2 = np.array([points_2d[3, 0] - points_2d[1, 0], points_2d[3, 1] - points_2d[1, 1], 0])
    coeff1 = np.linalg.norm(vector1)
    coeff2 = np.linalg.norm(vector2)
    vector1 = normalize_point(vector1)
    vector2 = normalize_point(vector2)
    centroid = np.array([points_2d[0, 0] + points_2d[3, 0], points_2d[0, 1] + points_2d[3, 1], float(z_max) + float(z_min)]) * 0.5
    basis = np.array([vector1, vector2, [0, 0, 1]])
    coeffs = np.array([coeff1, coeff2, z_max - z_min]) * 0.5
    return centroid, basis, coeffs


def project_3d_points_to_2d(points3d, R_ex, K):
    """
        Project 3d points from camera-centered coordinate to 2D image plane

        Parameters
        ----------
        points3d: numpy array
            3d location of point
        R_ex: numpy array
            extrinsic camera parameter
        K: numpy array
            intrinsci camera parameter
        Returns
        -------
        points2d: numpy array
            2d location of the point
    """
    points3d = points3d[:, [0, 2, 1]]
    points3d[:, 1] = -points3d[:, 1]
    points3d = R_ex.dot(points3d.T).T
    x3 = points3d[:, 0]
    y3 = points3d[:, 1]
    z3 = np.abs(points3d[:, 2])
    xx = x3 * K[0, 0] / z3 + K[0, 2]
    yy = y3 * K[1, 1] / z3 + K[1, 2]
    points2d = np.vstack((xx, yy))
    return points2d


def project_struct_bdb_to_2d(basis, coeffs, center, R_ex, K):
    """
        Project 3d bounding box to 2d bounding box

        Parameters
        ----------
        basis, coeffs, center, R_ex, K
            : K is the intrinsic camera parameter matrix
            : Rtilt is the extrinsic camera parameter matrix in right hand coordinates

        Returns
        -------
        bdb2d: dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
    """
    corners3d = get_corners_of_bb3d(basis, coeffs, center)
    corners = project_3d_points_to_2d(corners3d, R_ex, K)
    bdb2d = dict()
    bdb2d['x1'] = int(max(np.min(corners[0, :]), 1))  # x1
    bdb2d['y1'] = int(max(np.min(corners[1, :]), 1))  # y1
    bdb2d['x2'] = int(min(np.max(corners[0, :]), 2*K[0, 2]))  # x2
    bdb2d['y2'] = int(min(np.max(corners[1, :]), 2*K[1, 2]))  # y2
    if not check_bdb(bdb2d, 2*K[0, 2], 2*K[1, 2]):
        bdb2d = None
    return bdb2d


def check_bdb(bdb2d, m, n):
    """
        Check valid a bounding box is valid

        Parameters
        ----------
        bdb2d: dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        m: int
            width
        n: int
            height

        Returns
        -------
        valid: bool
    """
    if bdb2d['x1'] >= bdb2d['x2'] or bdb2d['y1'] >= bdb2d['y2'] or bdb2d['x1'] > m or bdb2d['y1'] > n:
        return False
    else:
        return True


def get_iou(bb1, bb2):
    """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_iou_cuboid(cu1, cu2):
    """
        Calculate the Intersection over Union (IoU) of two 3D cuboid.

        Parameters
        ----------
        cu1 : numpy array, 8x3
        cu2 : numpy array, 8x3

        Returns
        -------
        float
            in [0, 1]
    """
    polygon_1 = Polygon([(cu1[0][0], cu1[0][1]), (cu1[1][0], cu1[1][1]), (cu1[2][0], cu1[2][1]), (cu1[3][0], cu1[3][1])])
    polygon_2 = Polygon([(cu2[0][0], cu2[0][1]), (cu2[1][0], cu2[1][1]), (cu2[2][0], cu2[2][1]), (cu2[3][0], cu2[3][1])])
    intersect_2d = polygon_1.intersection(polygon_2).area
    inter_vol = intersect_2d * max(0.0, min(cu1[0][2], cu2[0][2]) - max(cu1[4][2], cu2[4][2]))
    vol1 = polygon_1.area * (cu1[0][2] - cu1[4][2])
    vol2 = polygon_2.area * (cu2[0][2] - cu2[4][2])
    return inter_vol / (vol1 + vol2 - inter_vol)

