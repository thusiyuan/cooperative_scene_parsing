"""
Created on April, 2018

@author: Siyuan Huang

configuration of the project

"""

import errno
import logging
import os
import pickle


class Config(object):
    def __init__(self, dataset='sunrgbd'):
        """
        Configuration of data paths.
        """
        self.dataset = dataset
        self.project_root = '/home/siyuan/Dropbox/Project/nips2018'
        if self.dataset == 'sunrgbd':
            self.metadata_root = '/home/siyuan/Documents/nips2018/sunrgbd'
            self.proposal_root = os.path.join(self.metadata_root, 'sunrgbdproposals')
            self.obj_category = ['recycle_bin', 'cpu', 'paper', 'toilet', 'stool', 'whiteboard', 'coffee_table',
                               'picture',
                               'keyboard', 'dresser', 'painting', 'bookshelf', 'night_stand', 'endtable', 'drawer',
                               'sink',
                               'monitor', 'computer', 'cabinet', 'shelf', 'lamp', 'garbage_bin', 'box', 'bed', 'sofa',
                               'sofa_chair', 'pillow', 'desk', 'table', 'chair']

    def bins(self):
        bin = dict()
        if self.dataset == 'sunrgbd':
            # center bins
            NUM_CENTER_BIN = 6
            X_WIDTH = 1.0
            Y_WIDTH = 1.0
            Z_WIDTH = 0.5
            bin['x_bin'] = [[(i - NUM_CENTER_BIN / 2) * X_WIDTH, (i - NUM_CENTER_BIN / 2 + 1) * X_WIDTH] for i in
                            range(NUM_CENTER_BIN)]
            bin['y_bin'] = [[i * Y_WIDTH, (i + 1) * Y_WIDTH] for i in range(NUM_CENTER_BIN)]
            bin['z_bin'] = [[(i - NUM_CENTER_BIN / 2) * Z_WIDTH, (i - NUM_CENTER_BIN / 2 + 1) * Z_WIDTH] for i in
                            range(NUM_CENTER_BIN)]
            DEPTH_WIDTH = 1.0
            bin['centroid_bin'] = [[i * DEPTH_WIDTH, (i + 1) * DEPTH_WIDTH] for i in
                            range(NUM_CENTER_BIN)]
            # yaw bins
            YAW_NUMBER_BINS = 2
            YAW_WIDTH = 20.0
            ROLL_NUMBER_BINS = 2
            ROLL_WIDTH = 40.0
            # yaw_bin = [[-20, 0], [0, 20]]
            bin['yaw_bin'] = [[-20.0 + i * YAW_WIDTH, -20.0 + (i + 1) * YAW_WIDTH] for i in range(YAW_NUMBER_BINS)]
            # roll_bin = [[-60, -20], [-20, 20]]
            bin['roll_bin'] = [[-60.0 + i * ROLL_WIDTH, -60.0 + (i + 1) * ROLL_WIDTH] for i in range(ROLL_NUMBER_BINS)]
            # orientation bin
            NUM_ORI_BIN = 6
            ORI_BIN_WIDTH = float(360 / NUM_ORI_BIN)
            bin['ori_bin'] = [[(i - NUM_ORI_BIN / 2) * ORI_BIN_WIDTH, (i - NUM_ORI_BIN / 2 + 1) * ORI_BIN_WIDTH] for i
                              in range(NUM_ORI_BIN)]
            # bin['layout_centroid_avg'] = [2.0, 1.6153291, 1.33276726]
            bin['layout_centroid_avg'] = [0.0, 1.6153291, 1.33276726]
            bin['layout_coeffs_avg'] = [2.21495791, 1.9857777, 2.62160042]
            bin['layout_normalize'] = [1.0, 1.0, 1.0]
            NUM_LAYOUT_ORI_BIN = 3
            ORI_LAYOUT_BIN_WIDTH = float(180 / NUM_LAYOUT_ORI_BIN)
            bin['layout_ori_bin'] = [[i * ORI_LAYOUT_BIN_WIDTH, (i + 1) * ORI_LAYOUT_BIN_WIDTH] for i in
                                     range(NUM_LAYOUT_ORI_BIN)]
            template_path = os.path.join(self.metadata_root, 'size_avg_category.pickle')
            avg_size = pickle.load(open(template_path, 'r'))
            bin['avg_size'] = [avg_size[obj] for obj in self.obj_category]
        elif self.dataset == 'suncg':
            # center bins
            NUM_CENTER_BIN = 6
            X_WIDTH = 1.0
            Y_WIDTH = 1.0
            Z_WIDTH = 0.5
            bin['x_bin'] = [[(i - NUM_CENTER_BIN / 2) * X_WIDTH, (i - NUM_CENTER_BIN / 2 + 1) * X_WIDTH] for i in
                            range(NUM_CENTER_BIN)]
            bin['y_bin'] = [[i * Y_WIDTH, (i + 1) * Y_WIDTH] for i in range(NUM_CENTER_BIN)]
            bin['z_bin'] = [[(i - NUM_CENTER_BIN / 2) * Z_WIDTH, (i - NUM_CENTER_BIN / 2 + 1) * Z_WIDTH] for i in
                            range(NUM_CENTER_BIN)]
            DEPTH_WIDTH = 1.0
            bin['centroid_bin'] = [[i * DEPTH_WIDTH, (i + 1) * DEPTH_WIDTH] for i in
                            range(NUM_CENTER_BIN)]
            YAW_NUMBER_BINS = 2
            YAW_WIDTH = 3.0
            ROLL_NUMBER_BINS = 2
            ROLL_WIDTH = 1.0
            # yaw_bin = [[-3, 0], [0, 3]]
            bin['yaw_bin'] = [[-3.0 + i * YAW_WIDTH, -3.0 + (i + 1) * YAW_WIDTH] for i in range(YAW_NUMBER_BINS)]
            # roll_bin = [[-12, -11], [-11, -10]]
            bin['roll_bin'] = [[-12.0 + i * ROLL_WIDTH, -12.0 + (i + 1) * ROLL_WIDTH] for i in range(ROLL_NUMBER_BINS)]
            # orientation bin
            NUM_ORI_BIN = 6
            ORI_BIN_WIDTH = float(360 / NUM_ORI_BIN)
            bin['ori_bin'] = [[(i - NUM_ORI_BIN / 2) * ORI_BIN_WIDTH, (i - NUM_ORI_BIN / 2 + 1) * ORI_BIN_WIDTH] for i
                              in range(NUM_ORI_BIN)]
            NUM_LAYOUT_ORI_BIN = 3
            ORI_LAYOUT_BIN_WIDTH = float(180 / NUM_LAYOUT_ORI_BIN)
            bin['layout_ori_bin'] = [[i * ORI_LAYOUT_BIN_WIDTH, (i + 1) * ORI_LAYOUT_BIN_WIDTH] for i in
                                     range(NUM_LAYOUT_ORI_BIN)]
            bin['layout_centroid_avg'] = [-0.33775898, 1.18361065, -0.11754291]
            # bin['layout_centroid_avg'] = [2, 2, 2]
            bin['layout_coeffs_avg'] = [2.92935682, 2.93277507, 1.43180995]
            bin['layout_normalize'] = [2.0, 2.0, 2.0]
            template_path = os.path.join(self.metadata_root, 'size_avg_category.pickle')
            avg_size = pickle.load(open(template_path, 'r'))
            bin['avg_size'] = [avg_size[obj] for obj in self.obj_category]
        return bin


def set_logger(name='learner.log'):
    if not os.path.exists(os.path.dirname(name)):
        try:
            os.makedirs(os.path.dirname(name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(name, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                                "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger


