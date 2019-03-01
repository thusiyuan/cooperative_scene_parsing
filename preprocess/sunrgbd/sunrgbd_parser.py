"""
Created on Mar, 2018

@author: Siyuan Huang

class of SUNRGBD data and data reader
"""
import os
import numpy as np
import pickle
from PIL import Image
from scipy.io import loadmat
import config
import sunrgbd_visualize
PATH = config.Config('sunrgbd')


# class of SUNRGBD Data
class SUNRGBDData(object):
    def __init__(self, K, R_ex, R_tilt, bdb2d, bdb3d, gt3dcorner, imgdepth, imgrgb, seg2d, sequence_name, sequence_id, scene_type):
        self._K = K
        # R_ex.T is the left-hand camera coordinates -> world coordinates transformation P_world = R_ex*P_camera
        self._R_ex = R_ex
        # R_tilt is the right-hand camera coordinates  -> world coordinates transformation P_world = R_tilt*P_camera(after transformed to x, z, -y)
        self._R_tilt = R_tilt
        self._bdb2d = bdb2d
        self._bdb3d = bdb3d
        self._gt3dcorner = gt3dcorner
        self._imgdepth = imgdepth
        self._imgrgb = imgrgb
        self._seg2d = seg2d
        self._sequence_name = sequence_name
        self._sequence_id = sequence_id
        self._height, self._width = np.shape(self._imgrgb)[:2]
        self._scene_type = scene_type

    def __str__(self):
        return 'sequence_name: {}, sequence_id: {}'.format(self._sequence_name, self._sequence_id)

    def __repr__(self):
        return self.__str__()

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def K(self):
        return self._K

    @property
    def R_ex(self):
        return self._R_ex

    @property
    def R_tilt(self):
        return self._R_tilt

    @property
    def bdb2d(self):
        return self._bdb2d

    @property
    def bdb3d(self):
        return self._bdb3d

    @property
    def gt3dcorner(self):
        return self._gt3dcorner

    @property
    def imgdepth(self):
        return self._imgdepth

    @property
    def imgrgb(self):
        return self._imgrgb

    @property
    def seg2d(self):
        return self._seg2d

    @property
    def sequence_name(self):
        return self._sequence_name

    @property
    def sequence_id(self):
        return self._sequence_id

    @property
    def scene_type(self):
        return self._scene_type


# load the ground truth for SUNRGBD Frame
def readsunrgbdframe(image_name=None, image_id=None):
    root = config.Config('sunrgbd')
    clean_data_path = root.clean_data_root
    with open(os.path.join(clean_data_path, 'imagelist.txt'), 'r') as f:
        image_list = [line.replace('\n', '') for line in f]
    f.close()
    if image_name:
        image_id = image_list.index(image_name) + 1
    with open(os.path.join(clean_data_path, 'data_all', str(image_id) + '.pickle'), 'r') as f:
        img_info = pickle.load(f)
    f.close()

    # change data root manually
    img_info['imgrgb_path'] = img_info['imgrgb_path'].replace('/home/siyuan/Documents/Dataset/SUNRGBD_ALL/', PATH.metadata_root + '/Dataset/')
    img_info['imgdepth_path'] = img_info['imgdepth_path'].replace('/home/siyuan/Documents/Dataset/SUNRGBD_ALL/', PATH.metadata_root + '/Dataset/')
    # load rgb img
    img_info['imgrgb'] = np.array(Image.open(img_info['imgrgb_path']))

    # load depth img
    imgdepth = np.array(Image.open(img_info['imgdepth_path'])).astype('uint16')
    imgdepth = (imgdepth >> 3) | (imgdepth << 13)
    imgdepth = imgdepth.astype('single') / 1000
    imgdepth[imgdepth > 8] = 8
    img_info['imgdepth'] = imgdepth

    if 'gt3dcorner' not in img_info.keys():
        img_info['gt3dcorner'] = None
    # load segmentation
    # img_info['seg2d'] = loadmat(img_info['seg2d_path'])['seg_2d']
    scene_category_path = os.path.join(root.data_root, img_info['sequence_name'], 'scene.txt')
    if not os.path.exists(scene_category_path):
        scene_category = None
    else:
        with open(scene_category_path, 'r') as f:
            scene_category = f.readline()
    data_frame = SUNRGBDData(img_info['K'], img_info['R_ex'], img_info['R_tilt'], img_info['bdb2d'], img_info['bdb3d'], img_info['gt3dcorner'], img_info['imgdepth'], img_info['imgrgb'], img_info['seg2d'], img_info['sequence_name'], image_id, scene_category)
    return data_frame


def demo():
    # load data
    # data_frame = readsunrgbdframe(image_id=6000)
    data_frame = readsunrgbdframe(image_id=0)
    print data_frame.bdb3d
    print data_frame.bdb2d
    # show rgbimg
    # plt.figure()
    # plt.imshow(data_frame.imgrgb)
    # plt.show()
    # print data_frame.K
    # print data_frame.R_ex
    # print np.arccos(data_frame.R_ex[1, 1])
    # print np.arcsin(data_frame.R_ex[2, 1])
    # get 2d corner
    # sunrgbd_visualize.show_2dcorner(data_frame.imgrgb, data_frame.gt3dcorner, data_frame.K, data_frame.R_ex, data_frame.R_tilt, img_only=0)

    # # show original 3dpoint
    # sunrgbd_visualize.show_3dpointcloud(data_frame.imgrgb, data_frame.imgdepth, data_frame.K, 40)
    #
    # # show 3d aligned points
    # sunrgbd_visualize.show_3dpointcloud_aligned(data_frame.imgrgb, data_frame.imgdepth, data_frame.K, data_frame.R_tilt, 40)
    #
    # # show depthimg
    # plt.figure()
    # im = plt.imshow(data_frame.imgdepth)
    # plt.show()
    #
    # # show segmentation
    # im = plt.imshow(data_frame.seg2d)
    # plt.show()
    #
    # # show 2d bounding boxes
    # sunrgbd_visualize.show_2dboxes(data_frame.imgrgb, data_frame.bdb2d)


def main():
    demo()


if __name__ == '__main__':
    main()
