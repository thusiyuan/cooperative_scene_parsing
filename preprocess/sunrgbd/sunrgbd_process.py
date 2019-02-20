"""
Created on Mar, 2018

@author: Siyuan Huang

Preprocess the SUNRGBD dataset
"""

import sys
sys.path.append('.')
import json
import pickle
import config
import os
import os.path as op
import numpy as np
from sunrgbd_parser import readsunrgbdframe
from utils.sunrgbd_utils import project_struct_bdb_to_2d, get_iou, check_bdb, get_corners_of_bb3d, get_bdb_from_corners, get_corners_of_bb3d_no_index, project_3d_points_to_2d
from utils.camera_utils import yaw_pitch_row_from_r, get_rotation_matrix_from_yaw_roll
from utils.vis_utils import show_2dboxes
from sklearn.cluster import KMeans
from scipy.io import loadmat
import random
from PIL import Image
import matplotlib.pyplot as plt
from scipy.misc import imresize

PATH = config.Config('sunrgbd')

OBJ_CATEGORY = ['bed', 'night_stand', 'ottoman', 'dresser_mirror', 'dresser', 'lamp', 'sofa', 'pillow', 'cabinet', 'sofa_chair',
                'chair', 'table', 'computer', 'monitor', 'tv', 'bag', 'tv_stand', 'box', 'bottle', 'book',
                'coffee_table', 'laptop', 'shelf', 'plant', 'desk', 'back_pack', 'speaker', 'endtable', 'fridge',
                'recycle_bin', 'garbage_bin', 'person', 'bench', 'printer', 'counter', 'toilet', 'sink', 'bathtub',
                'towel', 'door', 'painting', 'drawer', 'cup', 'island', 'basket', 'bookshelf', 'rack', 'stack_of_chairs',
                'stove', 'microwave', 'cubby', 'flower_vase', 'mouse', 'keyboard', 'paper', 'scanner', 'poster',
                'dining_table', 'bulletin_board', 'bowl', 'pot', 'books', 'whiteboard', 'projector', 'plate',
                'organizer', 'picture', 'telephone', 'switch', 'cart', 'mug', 'cpu', 'stool', 'tissue', 'mirror',
                'suits_case', 'thermos', 'tray', 'curtain', 'electric_fan', 'board', 'clock', 'blanket', 'podium',
                'urinal', 'cloth', 'glass', 'desktop', 'blinds', 'blackboard', 'container', 'oven', 'machine']

OBJ_CATEGORY_CLEAN = ['recycle_bin', 'cpu', 'paper', 'toilet', 'stool', 'whiteboard', 'coffee_table', 'picture',
                      'keyboard', 'dresser', 'painting', 'bookshelf', 'night_stand', 'endtable', 'drawer', 'sink',
                      'monitor', 'computer', 'cabinet', 'shelf', 'lamp', 'garbage_bin', 'box', 'bed', 'sofa',
                      'sofa_chair', 'pillow', 'desk', 'table', 'chair']

OBJ_CATEGORY_TEST = ['recycle_bin', 'cpu', 'paper', 'toilet', 'stool', 'whiteboard', 'coffee_table',
                      'keyboard', 'dresser', 'bookshelf', 'night_stand', 'endtable', 'sink',
                      'monitor', 'computer', 'cabinet', 'shelf', 'lamp', 'garbage_bin', 'box', 'bed', 'sofa',
                      'sofa_chair', 'desk', 'table', 'chair']


LAMBDA_SIZE = 10


def prepare_data(gt_2d_bdb=False, patch_h=224, patch_w=224, shift=True, iou_threshold=0.1):
    """
        Generating the ground truth for end-to-end training

        Parameters
        ----------
        gt_2d_bdb : bool
            indicates whether to use the ground truth of 2D bounding boxes
        patch_h: int
            the height of target resized patch
        patch_w: int
            the width of target resized potch
        iou_threshold : float
            iou threshold for two 2D bounding boxes
    """
    bin = PATH.bins()
    data_root = op.join(PATH.metadata_root, 'sunrgbd_train_test_data')
    train_path = list()
    test_path = list()
    layout_centroid = list()
    layout_coeffs = list()
    # obj_category = dict()
    if not op.exists(data_root):
        os.mkdir(data_root)
    for i in range(10335):
        sequence = readsunrgbdframe(image_id=i+1)
        print i+1
        sequence._R_tilt = loadmat(op.join(PATH.metadata_root, 'updated_rtilt', str(i+1) + '.mat'))['r_tilt']
        # R_ex is cam to world
        sequence._R_ex = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).dot(sequence.R_tilt).dot(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
        K = sequence.K
        result = []
        for bdb2d in sequence.bdb2d:
            if check_bdb(bdb2d, 2*sequence.K[0, 2], 2*sequence.K[1, 2]):
                result.append(bdb2d)
            else:
                print 'ground truth not valid'
        sequence._bdb2d = result

        bdb2d_from_3d_list = []
        with open(op.join(PATH.metadata_root, '2dbdb', str(i + 1) + '.json'), 'r') as f:
            detected_bdbs = json.load(f)
        f.close()
        boxes = list()
        for bdb3d in sequence.bdb3d:
            center = bdb3d['centroid'][0]
            coeffs = bdb3d['coeffs'][0]
            basis = bdb3d['basis'].astype('float32')
            if bdb3d['classname'][0] not in OBJ_CATEGORY_CLEAN:
                continue
            bdb2d_from_3d = project_struct_bdb_to_2d(basis, coeffs, center, sequence.R_ex.T, K)
            projected_2d_center = project_3d_points_to_2d(center.reshape(1, 3), sequence.R_ex.T, K)
            if bdb2d_from_3d is None:
                print '%s not valid' % (bdb3d['classname'][0])
                continue
            bdb2d_from_3d['classname'] = bdb3d['classname'][0]
            bdb2d_from_3d_list.append(bdb2d_from_3d)
            if gt_2d_bdb is True:
                max_iou = 0
                iou_ind = -1
                for j, bdb2d in enumerate(sequence.bdb2d):
                    if bdb2d['classname'] == bdb3d['classname'][0]:
                        iou = get_iou(bdb2d_from_3d, bdb2d)
                        if iou > iou_threshold and iou > max_iou:
                            iou_ind = j
                            max_iou = iou
                if iou_ind >= 0:
                    if shift:
                        shifted_box = random_shift_2d_box(sequence.bdb2d[iou_ind])
                        boxes.append({'2dbdb': shifted_box, '3dbdb': bdb3d,
                                      'projected_2d_center': projected_2d_center})
                    else:
                        boxes.append({'2dbdb': sequence.bdb2d[iou_ind], '3dbdb': bdb3d, 'projected_2d_center': projected_2d_center})
            else:
                max_iou = 0
                iou_ind = -1
                max_bdb = dict()
                for j, bdb2d in enumerate(detected_bdbs):
                    if bdb2d['class'] == bdb3d['classname'][0]:
                        box = bdb2d['bbox']
                        box = {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}
                        iou = get_iou(bdb2d_from_3d, box)
                        if iou > iou_threshold and iou > max_iou:
                            iou_ind = j
                            max_iou = iou
                            box['score'] = bdb2d['score']
                            box['classname'] = bdb2d['class']
                            max_bdb = box
                if iou_ind >= 0:
                    # print max_iou, bdb2d_from_3d, detected_bdbs[iou_ind]
                    if shift:
                        shifted_box = random_shift_2d_box(max_bdb)
                        boxes.append({'2dbdb': shifted_box, '3dbdb': bdb3d, 'projected_2d_center': projected_2d_center})
                    else:
                        boxes.append({'2dbdb': max_bdb, '3dbdb': bdb3d, 'projected_2d_center': projected_2d_center})
        # print boxes
        camera = dict()
        camera_flip = dict()
        camera['yaw_cls'], camera['yaw_reg'], camera['roll_cls'], camera['roll_reg'] = camera_cls_reg(sequence.R_ex.T, bin)
        camera['K'] = sequence.K
        # flip the camera
        camera_flip['yaw_cls'], camera_flip['yaw_reg'], camera_flip['roll_cls'], camera_flip['roll_reg'] = camera_cls_reg(sequence.R_ex.T, bin, flip=True)
        camera_flip['K'] = sequence.K
        template_path = op.join(PATH.metadata_root, 'size_avg_category.pickle')
        layout_pts = loadmat(op.join(PATH.metadata_root, '3dlayout', str(i+1) + '.mat'))['manhattan_layout'].T
        l_centroid, l_basis, l_coeffs = get_bdb_from_corners(layout_pts)
        # print l_centroid
        layout_centroid.append(l_centroid)
        layout_coeffs.append(l_coeffs)
        layout = dict()
        layout['centroid_reg'] = layout_centroid_avg_residual(l_centroid, bin['layout_centroid_avg'], bin['layout_normalize'])
        layout['coeffs_reg'] = layout_size_avg_residual(l_coeffs, bin['layout_coeffs_avg'])
        layout['ori_cls'], layout['ori_reg'] = ori_cls_reg(l_basis[1, :], bin, layout=True)
        layout_flip = dict()
        layout_flip['centroid_reg'] = layout_centroid_avg_residual(l_centroid, bin['layout_centroid_avg'], bin['layout_normalize'], flip=True)
        layout_flip['coeffs_reg'] = layout_size_avg_residual(l_coeffs, bin['layout_coeffs_avg'])
        layout_flip['ori_cls'], layout_flip['ori_reg'] = ori_cls_reg(l_basis[1, :], bin, layout=True, flip=True)
        # print layout['ori_cls'], layout_flip['ori_cls']
        # clean the ground truth
        with open(template_path, 'r') as f:
            size_template = pickle.load(f)
        f.close()
        boxes_out = list()
        boxes_out_flip = list()
        for box in boxes:
            box_set = dict()
            # box_set['ori_cls'], box_set['ori_reg'] = ori_cls_reg(box['3dbdb']['orientation'])
            box_set['ori_cls'], box_set['ori_reg'] = ori_cls_reg(box['3dbdb']['basis'][1, :], bin)
            # print box['3dbdb']['basis']
            # print basis_from_ori(num_from_bins(bin['ori_bin'], box_set['ori_cls'], box_set['ori_reg']))
            box_set['size_reg'] = size_avg_residual(box['3dbdb']['coeffs'][0], size_template, box['2dbdb']['classname'])
            box_set['bdb3d'] = get_corners_of_bb3d_no_index(box['3dbdb']['basis'], box['3dbdb']['coeffs'][0], box['3dbdb']['centroid'][0])
            box_set['x_cls'], box_set['x_reg'], box_set['y_cls'], box_set['y_reg'], box_set['z_cls'], box_set['z_reg'] = centroid_cls_reg(box['3dbdb']['centroid'][0], bin)
            box_set['bdb_pos'] = [box['2dbdb']['x1'], box['2dbdb']['y1'], box['2dbdb']['x2'], box['2dbdb']['y2']]
            box_set['bdb2d'] = [box['2dbdb']['x1'] / float(K[0, 2]), box['2dbdb']['y1'] / float(K[1, 2]), box['2dbdb']['x2'] / float(K[0, 2]), box['2dbdb']['y2'] / float(K[1, 2])]
            box_set['centroid_cls'], box_set['centroid_reg'] = bin_cls_reg(bin['centroid_bin'], np.linalg.norm(box['3dbdb']['centroid'][0]))
            delta_2d = list()
            delta_2d.append(((box_set['bdb_pos'][0] + box_set['bdb_pos'][2]) / 2 - box['projected_2d_center'][0][0]) / (box_set['bdb_pos'][2] - box_set['bdb_pos'][0]))
            delta_2d.append(((box_set['bdb_pos'][1] + box_set['bdb_pos'][3]) / 2 - box['projected_2d_center'][1][0]) / (box_set['bdb_pos'][3] - box_set['bdb_pos'][1]))
            box_set['delta_2d'] = delta_2d
            box_set['size_cls'] = OBJ_CATEGORY_CLEAN.index(box['2dbdb']['classname'])
            # print box_set['size_cls']
            # print box['2dbdb']['classname']
            boxes_out.append(box_set)
            # print box_set['3dbdb']['classname'], box_set['ori_cls'], box_set['ori_reg'], box_set['size_reg'], box_set['size_cls'], box_set['size_reg']
            # flip the boxes
            box_set_flip = dict()
            # box_set_flip['ori_cls'], box_set_flip['ori_reg'] = ori_cls_reg(box['3dbdb']['orientation'], flip=True)
            box_set_flip['ori_cls'], box_set_flip['ori_reg'] = ori_cls_reg(box['3dbdb']['basis'][1, :], bin, flip=True)
            box_set_flip['size_reg'] = size_avg_residual(box['3dbdb']['coeffs'][0], size_template, box['2dbdb']['classname'])
            box_set_flip['x_cls'], box_set_flip['x_reg'], box_set_flip['y_cls'], box_set_flip['y_reg'], box_set_flip['z_cls'], box_set_flip['z_reg'] = centroid_cls_reg(box['3dbdb']['centroid'][0], bin, flip=True)
            box_set_flip['centroid_cls'], box_set_flip['centroid_reg'] = bin_cls_reg(bin['centroid_bin'], np.linalg.norm(box['3dbdb']['centroid'][0]))
            box_set_flip['bdb_pos'] = [int(2 * K[0, 2] - box['2dbdb']['x2']), box['2dbdb']['y1'], int(2 * K[0, 2] - box['2dbdb']['x1']), box['2dbdb']['y2']]
            box_set_flip['bdb2d'] = [int(2 * K[0, 2] - box['2dbdb']['x2']) / float(K[0, 2]), box['2dbdb']['y1'] / float(K[1, 2]),
                                       int(2 * K[0, 2] - box['2dbdb']['x1']) / float(K[0, 2]), box['2dbdb']['y2'] / float(K[1, 2])]
            box_set_flip['size_cls'] = OBJ_CATEGORY_CLEAN.index(box['2dbdb']['classname'])
            coeffs_flip = size_from_template(box_set_flip['size_reg'], size_template, OBJ_CATEGORY_CLEAN[box_set_flip['size_cls']])
            centroid_flip = np.array([num_from_bins(bin['x_bin'], box_set_flip['x_cls'], box_set_flip['x_reg']), num_from_bins(bin['y_bin'], box_set_flip['y_cls'], box_set_flip['y_reg']), num_from_bins(bin['z_bin'], box_set_flip['z_cls'], box_set_flip['z_reg'])])
            basis_flip = basis_from_ori(num_from_bins(bin['ori_bin'], box_set_flip['ori_cls'], box_set_flip['ori_reg']))
            box_set_flip['bdb3d'] = get_corners_of_bb3d(basis_flip, coeffs_flip, centroid_flip)
            delta_2d_flip = [- delta_2d[0], delta_2d[1]]
            box_set_flip['delta_2d'] = delta_2d_flip
            # print box_set['delta_2d'], box_set_flip['delta_2d']
            boxes_out_flip.append(box_set_flip)
        if len(boxes_out) == 0:
            continue
        data = dict()
        data['rgb_path'] = op.join(PATH.metadata_root, 'images', '%06d.jpg' % (i+1))
        data['boxes'] = list_of_dict_to_dict_of_list(boxes_out)
        data['camera'] = camera
        data['layout'] = layout
        data['sequence_id'] = i + 1
        # fliped data
        data_flip = dict()
        data_flip['rgb_path'] = op.join(PATH.metadata_root, 'images', '%06d_flip.jpg' % (i+1))
        # img_flip = Image.open(data['rgb_path']).transpose(Image.FLIP_LEFT_RIGHT)
        # img_flip.save(data_flip['rgb_path'])
        data_flip['boxes'] = list_of_dict_to_dict_of_list(boxes_out_flip)
        data_flip['camera'] = camera_flip
        data_flip['layout'] = layout_flip
        data_flip['sequence_id'] = i + 1
        if shift:
            save_path = op.join(PATH.metadata_root, 'sunrgbd_train_test_data', str(i+1) + '_shift_5' + '.pickle')
            save_path_flip = op.join(PATH.metadata_root, 'sunrgbd_train_test_data', str(i+1) + '_flip' + '_shift_5' + '.pickle')
        else:
            save_path = op.join(PATH.metadata_root, 'sunrgbd_train_test_data', str(i + 1) + '.pickle')
            save_path_flip = op.join(PATH.metadata_root, 'sunrgbd_train_test_data', str(i + 1) + '_flip' + '.pickle')
        if (i + 1) <= 5050:
            test_path.append(save_path)
        else:
            train_path.append(save_path)
        with open(save_path, 'w') as f:
            pickle.dump(data, f)
        f.close()
        with open(save_path_flip, 'w') as f:
            pickle.dump(data_flip, f)
        f.close()
    print np.array(layout_centroid).mean(axis=0)
    print np.array(layout_coeffs).mean(axis=0)
    if not shift:
        with open(op.join(PATH.metadata_root, 'train.json'), 'w') as f:
            json.dump(train_path, f)
        f.close()
        with open(op.join(PATH.metadata_root, 'test.json'), 'w') as f:
            json.dump(test_path, f)
        f.close()


def get_inference_sequence(file_path):
    """
    :param file_path: path of 2D bounding boxes
    :return:
    """
    with open(file_path + '.json', 'r') as f:
        detected_bdbs = json.load(f)
    f.close()
    boxes = list()
    for j, bdb2d in enumerate(detected_bdbs):
            box = bdb2d['bbox']
            box = {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}
            box['score'] = bdb2d['score']
            box['classname'] = bdb2d['class']
            boxes.append({'2dbdb': box})
    camera = dict()
    camera['K'] = np.array([[529.5, 0., 365.], [0, 529.5, 265.], [0, 0, 1]])
    boxes_out = list()
    for box in boxes:
        box_set = dict()
        box_set['bdb_pos'] = [box['2dbdb']['x1'], box['2dbdb']['y1'], box['2dbdb']['x2'], box['2dbdb']['y2']]
        if box['2dbdb']['classname'] not in OBJ_CATEGORY_TEST:
            continue
        box_set['size_cls'] = OBJ_CATEGORY_CLEAN.index(box['2dbdb']['classname'])
        boxes_out.append(box_set)
    data = dict()
    data['rgb_path'] = file_path + '.jpg'
    data['camera'] = camera
    data['boxes'] = list_of_dict_to_dict_of_list(boxes_out)
    data['sequence_id'] = int(file_path.split('/')[-1])
    return data


def random_shift_2d_box(box2d, shift_ratio=0.1):
    """
    shift the 2d box randomly
    :param
        box2d: dict of 2D box
        shift_ratio: ratio of random shifting
    :return:
        new 2d box
    """
    r = shift_ratio
    xmin, ymin, xmax, ymax = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']
    h = ymax - ymin
    w = xmax - xmin
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cx2 = cx + w * r * (np.random.random() * 2 - 1)
    cy2 = cy + h * r * (np.random.random() * 2 - 1)
    h2 = h * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    w2 = w * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2'] = cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0
    return box2d


def dict_sort(dic):
    """
        sort the value and re-index the keys
        Return the list of sorted keys
    """
    num = list()
    keys = dic.keys()
    for key, value in dic.items():
        num.append(value)
    index = sorted(range(len(num)), key=lambda k: num[k])
    sorted_key = [keys[i] for i in index]
    return sorted_key


def list_of_dict_to_dict_of_list(dic):
    """
        From a list of dict to a dict of list
        Each returned value is numpy array
    """
    new_dic = dict()
    keys = dic[0].keys()
    for key in keys:
        new_dic[key] = list()
        for di in dic:
            new_dic[key].append(di[key])
        new_dic[key] = np.array(new_dic[key])
    return new_dic


def centroid_cls_reg(centroid, bin, flip=False):
    """
        Generating the ground truth for object center

        Parameters
        ----------
        centroid : numpy array
            position of object center
        bin: dict
            the bin for generating the data
        flip: bool
            whether to flip the image

        Returns
        -------
        x_cls, x_reg, y_cls, y_reg, z_cls, z_reg,
    """
    x_bin = bin['x_bin']
    y_bin = bin['y_bin']
    z_bin = bin['z_bin']
    if flip is True:
        x_cls, x_reg = bin_cls_reg(x_bin, - centroid[0])
    else:
        x_cls, x_reg = bin_cls_reg(x_bin, centroid[0])
    y_cls, y_reg = bin_cls_reg(y_bin, centroid[1])
    z_cls, z_reg = bin_cls_reg(z_bin, centroid[2])
    return x_cls, x_reg, y_cls, y_reg, z_cls, z_reg


def camera_cls_reg(r_ex, bin, flip=False):
    """
        Generating the ground truth for camera parameter (yaw and roll)

        Parameters
        ----------
        r_ex : numpy matrix
            camera parameter matrix
        bin: dict
            the bin for generating the data
        flip: bool
            whether to flip

        Returns
        -------
        yaw_cls, yaw_reg, roll_cls, roll_reg
    """
    yaw_bin = bin['yaw_bin']
    roll_bin = bin['roll_bin']
    yaw, pitch, roll = yaw_pitch_row_from_r(r_ex)
    yaw = - yaw / np.pi * 180
    roll = - roll / np.pi * 180
    # print yaw, roll
    if flip is True:
        yaw = - yaw
    yaw_cls, yaw_reg = bin_cls_reg(yaw_bin, yaw)
    roll_cls, roll_reg = bin_cls_reg(roll_bin, roll)
    return yaw_cls, yaw_reg, roll_cls, roll_reg


def ori_cls_reg(orientation, bin, layout=False, flip=False):
    """
        Generating the ground truth for object orientation

        Parameters
        ----------
        orientation : numpy array
            orientation vector of the object
        bin: dict
            bin for generating the data
        flip: bool
            whether to flip
        layout: bool
            whether use the bin of layout
        Returns
        -------
        cls: one-hot vector
            indicates which bin the orientation belongs
        reg: float
            the distance to the template center
    """
    if layout:
        ori_bin = bin['layout_ori_bin']
    else:
        ori_bin = bin['ori_bin']
    angle = np.arctan2(orientation[0], orientation[1]) / np.pi * 180
    if flip is True:
        # make sure the return angle is in (-pi, pi)
        if angle > 0:
            angle = 180 - angle
        else:
            angle = - angle - 180
    cls, reg = bin_cls_reg(ori_bin, angle)
    return cls, reg


def size_cls_reg(coeffs, template_path):
    """
        Generating the ground truth for object size

        Parameters
        ----------
        coeffs : numpy array
            size of the object
        template_path: dir
            directory that save the templates
        Returns
        -------
        cls: one-hot vector
            indicates which template the size belongs
        reg: numpy array
            the distance array to the template center
    """
    with open(template_path, 'r') as f:
        templates = pickle.load(f)
    f.close()
    cls, reg = center_cls_reg(templates, coeffs)
    return cls, LAMBDA_SIZE * reg


def basis_from_ori(ori):
    """
    :param ori: float
            the orientation angle
    :return: basis: 3x3 numpy array
            the basis in 3D coordinates
    """
    ori = ori / 180 * np.pi
    basis = np.zeros((3, 3))
    basis[0, 0] = np.cos(ori)
    basis[0, 1] = - np.sin(ori)
    basis[1, 0] = np.sin(ori)
    basis[1, 1] = np.cos(ori)
    basis[2, 2] = 1
    return basis


def size_avg_residual(coeffs, size_template, obj_category):
    """
    :param coeffs: object size
    :param size_template: dictionary that saves the mean size of each category
    :param obj_category:
    :return: size residual ground truth normalized by the average size
    """
    avg_size = size_template
    size_residual = (coeffs - avg_size[obj_category]) / avg_size[obj_category]
    return size_residual


def layout_centroid_avg_residual(centroid, avg, normalize_term, flip=False):
    """
    get the residual of the centroid of layout
    :param centroid: layout centroid
    :param avg: layout centroid average
    :param flip: whether to flip
    :param normalize_term normalize the layout
    :return: regression value
    """
    if flip is False:
        reg = (centroid - avg) / normalize_term
    else:
        flip_centroid = np.array([-centroid[0], centroid[1], centroid[2]])
        reg = (flip_centroid - avg) / normalize_term
    return reg


def layout_size_avg_residual(coeffs, avg):
    """
    get the residual of the centroid of layout
    :param coeffs: layout coeffs
    :param avg: layout centroid average
    :return: regression value
    """
    reg = (coeffs - avg) / avg
    return reg


def size_from_template(reg, size_template, obj_category):
    """
    :param reg: 3x1 numpy array
            size residual
    :param size_template: dictionary that saves the mean size of each category
    :param obj_category:
    :return: computed size
    """
    avg_size = size_template
    size = reg * avg_size[obj_category] + avg_size[obj_category]
    return size


def bin_cls_reg(bins, loc):
    """
        Given bins and value, compute where the value locates and the distance to the center

        Parameters
        ----------
        bins : list
            The bins, eg. [[-x, 0], [0, x]]
        loc: float
            The location

        Returns
        -------
        cls: int
            indicates which bin is the location for classification
        reg: float
            the distance to the center of the corresponding bin
    """
    len_bin = bins[0][1] - bins[0][0]
    dist = ([float(abs(loc - float(bn[0] + bn[1]) / 2)) for bn in bins])
    min_anchor = dist.index(min(np.abs(dist)))
    cls = min_anchor
    reg = float(loc - float(bins[min_anchor][0] + bins[min_anchor][1]) / 2) / float(len_bin)
    return cls, reg


def num_from_bins(bins, cls, reg):
    """
    :param bins: list
        The bins
    :param cls: int
        Classification result
    :param reg:
        Regression result
    :return: computed value
    """
    bin_width = bins[0][1] - bins[0][0]
    bin_center = float(bins[cls][0] + bins[cls][1]) / 2
    return bin_center + reg * bin_width


def center_cls_reg(centers, loc):
    """
        Given centers and value, compute where the value locates and the distance to the center

        Parameters
        ----------
        centers : list
            The centers, eg. [-x, 0, x] or [[a_1, b_1], [a_2, b_2], [a_3, b_3]]
        loc: float or list
            The location

        Returns
        -------
        cls: int
            indicates which bin is the location for classification
        reg: float
            the distance to the center of the corresponding bin
    """
    dist = ([np.linalg.norm(np.array(loc - center)) for center in centers])
    min_anchor = dist.index(min(np.abs(dist)))
    cls = min_anchor
    reg = loc - centers[min_anchor]
    return cls, reg


def learn_size_bin(category_specific=True):
    """
        Learn the size distribution and generating size bins for regression

        Parameters
        ----------
        category_specific: bool
            indicates whether learn the size conditional on the category

        Returns
        -------
        size_bins: dict
            Keys: {'category'}
            The bins for each category
    """
    if category_specific:
        N_CLUSTER = 8
    else:
        N_CLUSTER = 16
    size_bins = dict()
    size_info = dict()
    for i in range(0, 10335):
        sequence = readsunrgbdframe(image_id=i + 1)
        for bdb3d in sequence.bdb3d:
            classname = bdb3d['classname'][0]
            coeffs = bdb3d['coeffs'][0]
            if classname not in size_info:
                size_info[classname] = list()
            size_info[classname].append(coeffs)
    if category_specific:
        size_avg = dict()
        for category in size_info.keys():
            size_array = np.array(size_info[category])
            mean_size = size_array.mean(axis=0)
            if category in OBJ_CATEGORY_CLEAN:
                size_avg[category] = mean_size
            m = size_array.shape[0]
            if m <= N_CLUSTER:
                kmeans = KMeans(n_clusters=int(m/2)+1, random_state=0).fit(size_array)
            else:
                kmeans = KMeans(n_clusters=N_CLUSTER, random_state=0).fit(size_info[category])
            size_bins[category] = kmeans.cluster_centers_
            print category, size_bins[category]
        with open(op.join(PATH.metadata_root, 'size_bin_category.pickle'), 'w') as f:
            pickle.dump(size_bins, f)
        f.close()
        with open(op.join(PATH.metadata_root, 'size_avg_category.pickle'), 'w') as f:
            pickle.dump(size_avg, f)
        f.close()
        print size_avg
    else:
        size_array = list()
        for key, value in size_info.items():
            size_array.extend(value)
        size_array = np.array(size_array)
        kmeans = KMeans(n_clusters=N_CLUSTER, random_state=0).fit(size_array)
        with open(op.join(PATH.metadata_root, 'size_bin_full.pickle'), 'w') as f:
            pickle.dump(kmeans.cluster_centers_, f)
        print kmeans.cluster_centers_


def main():
    prepare_data(False, shift=False)
    # learn_size_bin(category_specific=True)
    # learn_size_bin(category_specific=False)


if __name__ == '__main__':
    main()
