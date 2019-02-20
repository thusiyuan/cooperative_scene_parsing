"""
Created on April, 2018

@author: Siyuan Huang

pytorch network utils
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import utils.camera_utils
PI = 3.141592653589793


def conv(in_plane, out_plane, kernel_size, stride):
    """
    conv2d layer with same padding
    :param in_plane:
    :param out_plane:
    :param kernel_size:
    :return: conv with same padding
    """
    return nn.Sequential(
        nn.Conv2d(in_plane, out_plane, kernel_size=kernel_size, padding=(kernel_size - 1) // stride, stride=stride),
    )


def basis_from_ori(ori):
    """
    :param ori: torch tensor
            the orientation angle
    :return: basis: 3x3 tensor
            the basis in 3D coordinates
    """
    n = ori.size(0)
    ori = ori / 180.0 * PI
    basis = torch.zeros((n, 3, 3)).cuda()
    basis[:, 0, 0] = torch.cos(ori)
    basis[:, 0, 1] = - torch.sin(ori)
    basis[:, 1, 0] = torch.sin(ori)
    basis[:, 1, 1] = torch.cos(ori)
    basis[:, 2, 2] = 1
    return basis


def get_corners_of_bb3d(basis, coeffs, centroid):
    """
    :param basis: b x 3 x 3 tensor
    :param coeffs: b x 3 tensor
    :param centroid:  b x 3 tensor
    :return: corners b x 8 x 3 tensor
    """
    n = basis.size(0)
    corners = torch.zeros((n, 8, 3)).cuda()
    coeffs = coeffs.view(n, 3, 1).expand(-1, -1, 3)
    centroid = centroid.view(n, 1, 3).expand(-1, 8, -1)
    corners[:, 0, :] = -basis[:, 0, :] * coeffs[:, 0, :] + basis[:, 1, :] * coeffs[:, 1, :] + basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 1, :] = basis[:, 0, :] * coeffs[:, 0, :] + basis[:, 1, :] * coeffs[:, 1, :] + basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 2, :] = basis[:, 0, :] * coeffs[:, 0, :] + -basis[:, 1, :] * coeffs[:, 1, :] + basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 3, :] = -basis[:, 0, :] * coeffs[:, 0, :] + -basis[:, 1, :] * coeffs[:, 1, :] + basis[:, 2, :] * coeffs[:, 2, :]

    corners[:, 4, :] = -basis[:, 0, :] * coeffs[:, 0, :] + basis[:, 1, :] * coeffs[:, 1, :] + -basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 5, :] = basis[:, 0, :] * coeffs[:, 0, :] + basis[:, 1, :] * coeffs[:, 1, :] + -basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 6, :] = basis[:, 0, :] * coeffs[:, 0, :] + -basis[:, 1, :] * coeffs[:, 1, :] + -basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 7, :] = -basis[:, 0, :] * coeffs[:, 0, :] + -basis[:, 1, :] * coeffs[:, 1, :] + -basis[:, 2, :] * coeffs[:, 2, :]
    corners = corners + centroid
    return corners


def to_dict_tensor(dicts, if_cuda):
    dicts_new = copy.copy(dicts)
    for key, value in dicts_new.items():
        value_new = torch.from_numpy(np.array(value))
        if value_new.type() == 'torch.DoubleTensor':
            value_new = value_new.float()
        if if_cuda:
            value_new = value_new.cuda()
        dicts_new[key] = value_new
    return dicts_new


def physical_violation(bdb_layout, bdb_3d):
    """
    compute the loss of physical violation
    :param bdb_layout: 1 x 8 x 3 tensor
    :param bdb_3d: b x 8 x 3 tensor
    :return:
    """
    b = bdb_3d.size(0)
    layout_max = torch.max(bdb_layout, dim=1)[0].expand(b, -1)  # bx3
    layout_min = torch.min(bdb_layout, dim=1)[0].expand(b, -1)  # bx3
    bdb_max = torch.max(bdb_3d, dim=1)[0]  # b x 3
    bdb_min = torch.min(bdb_3d, dim=1)[0]  # b x 3
    violation = torch.nn.functional.relu(bdb_max - layout_max) + torch.nn.functional.relu(layout_min - bdb_min) # b x 3
    return violation, torch.zeros(b, 3).cuda()


def get_rotation_matix_result(bins_tensor, yaw_cls_gt, yaw_reg_result, roll_cls_gt, roll_reg_result):
    """
    :param bins_tensor:
    :param yaw_cls_gt:
    :param yaw_reg_result:
    :param roll_cls_gt:
    :param roll_reg_result:
    :return: r_ex 1 x 3 x 3 tensor
    """
    yaw_result = torch.gather(yaw_reg_result, 1, yaw_cls_gt.view(yaw_cls_gt.size(0), 1).expand(yaw_cls_gt.size(0), 1)).squeeze(1)
    roll_result = torch.gather(roll_reg_result, 1, roll_cls_gt.view(roll_cls_gt.size(0), 1).expand(roll_cls_gt.size(0), 1)).squeeze(1)
    yaw = num_from_bins(bins_tensor['yaw_bin'], yaw_cls_gt, yaw_result)
    roll = num_from_bins(bins_tensor['roll_bin'], roll_cls_gt, roll_result)
    r_ex = get_rotation_matrix_from_yaw_roll(-yaw / 180.0 * PI, -roll / 180.0 * PI)
    return r_ex


def get_rotation_matrix_gt(bins_tensor, yaw_cls_gt, yaw_reg_gt, roll_cls_gt, roll_reg_gt):
    yaw = num_from_bins(bins_tensor['yaw_bin'], yaw_cls_gt, yaw_reg_gt)
    roll = num_from_bins(bins_tensor['roll_bin'], roll_cls_gt, roll_reg_gt)
    r_ex = get_rotation_matrix_from_yaw_roll(- yaw / 180.0 * PI, - roll / 180.0 * PI)
    return r_ex


def get_bdb_3d_result(bins_tensor, ori_cls_gt, ori_reg_result, centroid_cls_gt, centroid_reg_result, size_cls_gt, size_reg_result, P, K, R_ex):
    # coeffs
    size_reg = torch.gather(size_reg_result, 1, size_cls_gt.view(size_cls_gt.size(0), 1, 1).expand(size_cls_gt.size(0), 1, size_reg_result.size(2))).squeeze(1)
    coeffs = (size_reg + 1) * bins_tensor['avg_size'][size_cls_gt, :]   # b x 3
    # centroid
    centroid_reg = torch.gather(centroid_reg_result, 1, centroid_cls_gt.view(centroid_cls_gt.size(0), 1).expand(centroid_cls_gt.size(0), 1)).squeeze(1)
    centroid_depth = num_from_bins(bins_tensor['centroid_bin'], centroid_cls_gt, centroid_reg)
    centroid = rgbd_to_world(P, centroid_depth, K, R_ex)  # b x 3

    # basis
    ori_reg = torch.gather(ori_reg_result, 1, ori_cls_gt.view(ori_cls_gt.size(0), 1).expand(ori_cls_gt.size(0), 1)).squeeze(1)
    ori = num_from_bins(bins_tensor['ori_bin'], ori_cls_gt, ori_reg)
    basis = basis_from_ori(ori)
    bdb = get_corners_of_bb3d(basis, coeffs, centroid)
    return bdb


def get_bdb_3d_gt(bins_tensor, ori_cls_gt, ori_reg_gt, centroid_cls_gt, centroid_reg_gt, size_cls_gt, size_reg_gt, P, K, R_ex):
    # coeffs
    coeffs = (size_reg_gt + 1) * bins_tensor['avg_size'][size_cls_gt, :]   # b x 3
    # centroid
    centroid_depth = num_from_bins(bins_tensor['centroid_bin'], centroid_cls_gt, centroid_reg_gt)
    centroid = rgbd_to_world(P, centroid_depth, K, R_ex)   # b x 3
    # basis
    ori = num_from_bins(bins_tensor['ori_bin'], ori_cls_gt, ori_reg_gt)
    basis = basis_from_ori(ori)
    bdb = get_corners_of_bb3d(basis, coeffs, centroid)
    return bdb


def rgbd_to_world(p, depth, K, R_ex):
    """
    Given pixel location and depth, get world coordinates
    :param p: b x 2 tensor
    :param depth: b tensor
    :param k: 1 x 3 x 3 tensor
    :param r_ex: 1 x 3 x 3 tensor
    :return: p_world_right: b x 3 tensor in right hand coordinate
    """
    n = p.size(0)
    x_temp = (p[:, 0] + 1 - K[0, 0, 2]) / K[0, 0, 0]
    y_temp = (p[:, 1] + 1 - K[0, 1, 2]) / K[0, 1, 1]
    z_temp = 1
    x = x_temp / torch.sqrt(x_temp**2 + y_temp**2 + z_temp**2) * depth
    y = y_temp / torch.sqrt(x_temp**2 + y_temp**2 + z_temp**2) * depth
    z = z_temp / torch.sqrt(x_temp**2 + y_temp**2 + z_temp**2) * depth
    p_cam = torch.stack((x, y, z), 1).view(n, 3, 1) # n x 3
    p_world = torch.bmm(torch.transpose(R_ex, 1, 2).expand(n, -1, -1), p_cam)
    p_world_right = torch.stack((p_world[:, 0, 0], p_world[:, 2, 0], -p_world[:, 1, 0]), 1)
    return p_world_right


def get_layout_bdb(bins_tensor, ori_cls, ori_reg, centroid_reg, coeffs_reg):
    """
    get the eight corners of 3D bounding box
    :param bins_tensor:
    :param ori_cls: 1 tensor
    :param ori_reg: 1 x 3 tensor
    :param centroid_reg: 1 x 3 tensor
    :param coeffs_reg: 1 x 3 tensor
    :return: bdb
    """
    ori_reg = torch.gather(ori_reg, 1, ori_cls.view(1, 1).expand(1, 1)).squeeze(1)
    ori = num_from_bins(bins_tensor['layout_ori_bin'], ori_cls, ori_reg)
    basis = basis_from_ori(ori)
    coeffs_reg = (coeffs_reg + 1) * bins_tensor['layout_coeffs_avg']
    centroid_reg = centroid_reg * bins_tensor['layout_normalize'] + bins_tensor['layout_centroid_avg']
    bdb = get_corners_of_bb3d(basis, coeffs_reg, centroid_reg)
    return bdb


def get_bdb_2d_result(bdb3d, r_ex, K):
    """
    :param bins_tensor:
    :param bdb3d: b x 8 x 3 tensor
    :param r_ex:
    :param K: 1 x 3 x 3 tensor
    :return:
    """
    n = bdb3d.size(0)
    points_2d = project_3d_points_to_2d(bdb3d, r_ex, K)  # b x 8 x 2
    x1 = torch.max(torch.min(points_2d[:, :, 0], dim=1)[0], torch.ones(n).cuda()) / (K[0, 0, 2].float())
    y1 = torch.max(torch.min(points_2d[:, :, 1], dim=1)[0], torch.ones(n).cuda()) / (K[0, 1, 2].float())
    x2 = torch.min(torch.max(points_2d[:, :, 0], dim=1)[0], torch.ones(n).cuda() * 2 * K[0, 0, 2]) / (K[0, 0, 2].float())
    y2 = torch.min(torch.max(points_2d[:, :, 1], dim=1)[0], torch.ones(n).cuda() * 2 * K[0, 1, 2]) / (K[0, 1, 2].float())
    return torch.stack((x1, y1, x2, y2), 1)


def get_bdb_2d_gt(bins_tensor, bdb3d, yaw_cls_gt, yaw_reg_gt, roll_cls_gt, roll_reg_gt, K):
    n = bdb3d.size(0)
    yaw = num_from_bins(bins_tensor['yaw_bin'], yaw_cls_gt, yaw_reg_gt)
    roll = num_from_bins(bins_tensor['roll_bin'], roll_cls_gt, roll_reg_gt)
    r_ex = get_rotation_matrix_from_yaw_roll(- yaw / 180.0 * PI, - roll / 180.0 * PI)
    points_2d = project_3d_points_to_2d(bdb3d, r_ex, K)  # b x 8 x 2
    x1 = torch.max(torch.min(points_2d[:, :, 0], dim=1)[0], torch.ones(n).cuda())
    y1 = torch.max(torch.min(points_2d[:, :, 1], dim=1)[0], torch.ones(n).cuda())
    x2 = torch.min(torch.max(points_2d[:, :, 0], dim=1)[0], torch.ones(n).cuda() * 2 * K[0, 0, 2])
    y2 = torch.min(torch.max(points_2d[:, :, 1], dim=1)[0], torch.ones(n).cuda() * 2 * K[0, 1, 2])
    return torch.stack((x1, y1, x2, y2), 1)


def project_3d_points_to_2d(points3d, R_ex, K):
    """
    project 3d points to 2d
    :param points3d: b x 8 x 3 tensor
    :param R_ex: 1 x 3 x 3 tensor
    :param K: 1 x 3 x 3 tensor
    :return:
    """
    n = points3d.size(0)
    points3d_left_hand = torch.stack((points3d[:, :, 0],  -points3d[:, :, 2], points3d[:, :, 1]), 2)  # transform to left hand coordinate
    points_cam_ori = torch.transpose(torch.bmm(R_ex.expand(n, -1, -1), torch.transpose(points3d_left_hand, 1, 2)), 1, 2)   # b x 8 x 3
    points_cam = torch.transpose(torch.stack((points_cam_ori[:, :, 0], points_cam_ori[:, :, 1], torch.abs(points_cam_ori[:, :, 2])), 2), 1, 2) # b x 3 x 8
    points_2d_ori = torch.transpose(torch.bmm(K.expand(n, -1, -1), points_cam), 1, 2)  # b x 8 x 3
    points_2d = torch.stack((points_2d_ori[:, :, 0] / points_2d_ori[:, :, 2], points_2d_ori[:, :, 1] / points_2d_ori[:, :, 2]), 2) # n x 8 x 2
    return points_2d


def get_rotation_matrix_from_yaw_roll(yaw, roll):
    """
    :param yaw: b x 1 tensor
    :param roll: b x 1 tensor
    :return: R_ex in left hand coordinates
    """
    n = yaw.size(0)
    R = torch.zeros((n, 3, 3)).cuda()
    R[:, 0, 0] = torch.cos(yaw)
    R[:, 0, 1] = - torch.sin(yaw) * torch.cos(roll)
    R[:, 0, 2] = torch.sin(roll) * torch.sin(yaw)
    R[:, 1, 0] = torch.sin(yaw)
    R[:, 1, 1] = torch.cos(roll) * torch.cos(yaw)
    R[:, 1, 2] = - torch.cos(yaw) * torch.sin(roll)
    R[:, 2, 0] = 0
    R[:, 2, 1] = torch.sin(roll)
    R[:, 2, 2] = torch.cos(roll)
    return R


def num_from_bins(bins, cls, reg):
    """
    :param bins: b x 2 tensors
    :param cls: b long tensors
    :param reg: b tensors
    :return: bin_center: b tensors
    """
    bin_width = (bins[0][1] - bins[0][0])
    bin_center = (bins[cls, 0] + bins[cls, 1]) / 2
    return bin_center + reg * bin_width


"""
Functions for testing
"""


def get_bdb_evaluation(bins_tensor, ori_cls_gt, ori_reg_result, centroid_cls_gt, centroid_reg_result, size_cls_gt, size_reg_result, P, K, R_ex):
    size_reg = torch.gather(size_reg_result, 1, size_cls_gt.view(size_cls_gt.size(0), 1, 1).expand(size_cls_gt.size(0), 1, size_reg_result.size(2))).squeeze(1)
    coeffs = (size_reg + 1) * bins_tensor['avg_size'][size_cls_gt, :]   # b x 3
    # centroid
    centroid_reg = torch.gather(centroid_reg_result, 1, centroid_cls_gt.view(centroid_cls_gt.size(0), 1).expand(centroid_cls_gt.size(0), 1)).squeeze(1)
    centroid_depth = num_from_bins(bins_tensor['centroid_bin'], centroid_cls_gt, centroid_reg)
    centroid = rgbd_to_world(P, centroid_depth, K, R_ex)  # b x 3
    # basis
    ori_reg = torch.gather(ori_reg_result, 1, ori_cls_gt.view(ori_cls_gt.size(0), 1).expand(ori_cls_gt.size(0), 1)).squeeze(1)
    ori = num_from_bins(bins_tensor['ori_bin'], ori_cls_gt, ori_reg)
    basis = basis_from_ori(ori)
    n = ori_cls_gt.size(0)
    bdb_output = [{'basis': basis[i, :, :].cpu().numpy(), 'coeffs': coeffs[i, :].cpu().numpy(), 'centroid': centroid[i, :].cpu().numpy(), 'classid':size_cls_gt[i].cpu().numpy() + 1} for i in range(n)]
    return bdb_output


def get_yaw_roll_error(bin, reg_result, cls_result, reg_gt, cls_gt):
    result = torch.gather(reg_result, 1, torch.argmax(cls_result, 1).view(cls_result.size(0), 1).expand(cls_result.size(0), 1)).squeeze(1)
    result_num = num_from_bins(bin, torch.argmax(cls_result), result)
    gt_num = num_from_bins(bin, cls_gt, reg_gt)
    return torch.abs(result_num - gt_num)

# basis = basis_from_ori(torch.rand(2))
# coeffs = torch.rand((2, 3))
# centroid = torch.rand((2, 3))
# get_corners_of_bb3d(basis, coeffs, centroid)


