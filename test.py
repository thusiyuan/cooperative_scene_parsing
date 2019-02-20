"""
Created on April, 2018

@author: Siyuan Huang

Test the SUNCG or SUNRGBD dataset
"""
import torch
import torch.nn as nn
import argparse
import os
import os.path as op
import torch.optim
from models.model_res import Bdb3dNet
from models.model_res import PosNet
from utils.net_utils import get_rotation_matix_result, get_rotation_matrix_gt, to_dict_tensor, get_bdb_2d_result, get_bdb_3d_result, get_bdb_3d_gt, get_layout_bdb, get_bdb_evaluation, physical_violation, get_yaw_roll_error
import config
from scipy.io import savemat
from utils.sunrgbd_utils import get_iou_cuboid, get_corners_of_bb3d, get_iou
import time
import numpy as np


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Training settings
parser = argparse.ArgumentParser(description='PyTorch 3D Network')
parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--metadataPath', type=str, default='/home/siyuan/Documents/nips2018', help='data saving dir')
parser.add_argument('--dataset', type=str, default='sunrgbd', help='sunrgbd or suncg. Default=sunrgbd')
parser.add_argument('--cls_reg_ratio', type=float, default=10, help='the ratio between the loss of classification and regression')
parser.add_argument('--obj_cam_ratio', type=float, default=1, help='the ratio between the loss of classification and regression')
parser.add_argument('--branch', type=str, default='jointnet', help='posenet, bdbnet or jointnet')
parser.add_argument('--model_path_pose', type=str, default='sunrgbd/models_final/joint_posenet_full.pth', help='the directory of trained model')
parser.add_argument('--model_path_bdb', type=str, default='sunrgbd/models_final/joint_bdbnet_full.pth')
parser.add_argument('--vis', type=bool, default=False, help='whether to visualize the result')
parser.add_argument('--save_result', type=bool, default=False, help='whether to save the result')

opt = parser.parse_args()
print opt

dataset_config = config.Config(opt.dataset)
bins_tensor = to_dict_tensor(dataset_config.bins(), if_cuda=opt.cuda)

device = torch.device("cuda" if opt.cuda else "cpu")
if opt.cuda and not torch.cuda.is_available():
    raise Exception('No GPU found, please run without --cuda')

torch.manual_seed(opt.seed)

print '======> loading dataset'

posenet = PosNet().to(device)
bdb3dnet = Bdb3dNet().to(device)

pretrained_path_pose = op.join(opt.metadataPath, opt.model_path_pose)
pretrained_path_bdb = op.join(opt.metadataPath, opt.model_path_bdb)
if opt.branch == 'posenet':
    posenet.load_weight(pretrained_path_pose)
if opt.branch == 'bdbnet':
    bdb3dnet.load_weight(pretrained_path_bdb)
if opt.branch == 'jointnet':
    posenet.load_weight(pretrained_path_pose)
    bdb3dnet.load_weight(pretrained_path_bdb)

from data.sunrgbd import sunrgbd_test_loader
test_loader = sunrgbd_test_loader(opt)

cls_criterion = nn.CrossEntropyLoss(size_average=True, reduce=True)
reg_criterion = nn.SmoothL1Loss(size_average=True, reduce=True)
mse_criterion = nn.MSELoss(size_average=True, reduce=True)

result_save_path = op.join(opt.metadataPath, opt.dataset, 'results_full')
if not op.exists(result_save_path):
    os.mkdir(result_save_path)

IoU3D = dict()
IoU2D = dict()
EVALUATE_OBJECT = ['recycle_bin', 'cpu', 'paper', 'toilet', 'stool', 'whiteboard', 'coffee_table', 'picture', 'keyboard', 'dresser', 'painting', 'bookshelf', 'night_stand', 'endtable', 'drawer', 'sink', 'monitor', 'computer', 'cabinet', 'shelf', 'lamp', 'garbage_bin', 'box', 'bed', 'sofa', 'sofa_chair', 'pillow', 'desk', 'table', 'chair']
OBJ_CATEGORY_CLEAN = ['recycle_bin', 'cpu', 'paper', 'toilet', 'stool', 'whiteboard', 'coffee_table', 'picture',
                      'keyboard', 'dresser', 'painting', 'bookshelf', 'night_stand', 'endtable', 'drawer', 'sink',
                      'monitor', 'computer', 'cabinet', 'shelf', 'lamp', 'garbage_bin', 'box', 'bed', 'sofa',
                      'sofa_chair', 'pillow', 'desk', 'table', 'chair']


def joint_loss(cls_result, cls_gt, reg_result, reg_gt):
    cls_loss = cls_criterion(cls_result, cls_gt)
    if len(reg_result.size()) == 3:
        reg_result = torch.gather(reg_result, 1, cls_gt.view(reg_gt.size(0), 1, 1).expand(reg_gt.size(0), 1, reg_gt.size(1)))
    else:
        reg_result = torch.gather(reg_result, 1, cls_gt.view(reg_gt.size(0), 1).expand(reg_gt.size(0), 1))
    reg_result = reg_result.squeeze(1)
    reg_loss = reg_criterion(reg_result, reg_gt)
    return cls_loss, opt.cls_reg_ratio * reg_loss


def joint_size_loss(reg_result, cls_gt, reg_gt):
    if len(reg_result.size()) == 3:
        reg_result = torch.gather(reg_result, 1, cls_gt.view(reg_gt.size(0), 1, 1).expand(reg_gt.size(0), 1, reg_gt.size(1)))
    else:
        reg_result = torch.gather(reg_result, 1, cls_gt.view(reg_gt.size(0), 1).expand(reg_gt.size(0), 1))
    reg_result = reg_result.squeeze(1)
    reg_loss = reg_criterion(reg_result, reg_gt)
    return opt.cls_reg_ratio * reg_loss


def test_epoch(epoch):
    total_loss_record = AverageMeter()
    yaw_reg_loss_record = AverageMeter()
    yaw_cls_loss_record = AverageMeter()
    roll_reg_loss_record = AverageMeter()
    roll_cls_loss_record = AverageMeter()
    size_reg_loss_record = AverageMeter()
    size_cls_loss_record = AverageMeter()
    ori_reg_loss_record = AverageMeter()
    ori_cls_loss_record = AverageMeter()
    centroid_reg_loss_record = AverageMeter()
    centroid_cls_loss_record = AverageMeter()
    corner_loss_record = AverageMeter()
    bdb_loss_record = AverageMeter()
    phy_loss_record = AverageMeter()
    lo_ori_reg_loss_record = AverageMeter()
    lo_ori_cls_loss_record = AverageMeter()
    lo_coeffs_loss_record = AverageMeter()
    lo_centroid_loss_record = AverageMeter()
    yaw_error_record = AverageMeter()
    roll_error_record = AverageMeter()
    offset_2d_loss_record = AverageMeter()
    if opt.branch == 'posenet' or opt.branch == 'jointnet':
        posenet.eval()
    if opt.branch == 'bdbnet' or opt.branch == 'jointnet':
        bdb3dnet.eval()
    with torch.no_grad():
        for i, sequence in enumerate(test_loader):
            sequence_id = sequence['sequence_id']
            if opt.branch == 'posenet' or opt.branch == 'jointnet':
                image = sequence['image'].to(device)
                K, yaw_reg, yaw_cls, roll_reg, roll_cls, lo_ori_reg, lo_ori_cls, lo_centroid, lo_coeffs = \
                    sequence['camera']['K'].float().to(device), \
                    sequence['camera']['yaw_reg'].float().to(device), \
                    sequence['camera']['yaw_cls'].long().to(device), \
                    sequence['camera']['roll_reg'].float().to(device), \
                    sequence['camera']['roll_cls'].long().to(device), \
                    sequence['layout']['ori_reg'].float().to(device), \
                    sequence['layout']['ori_cls'].long().to(device), \
                    sequence['layout']['centroid_reg'].float().to(device), \
                    sequence['layout']['coeffs_reg'].float().to(device)
                # feature = posenet.resnet(image)
                # np.save(op.join(result_save_path, 'feature', str(sequence_id[0].numpy())), feature.detach().cpu().numpy())
                yaw_reg_result, roll_reg_result, yaw_cls_result, roll_cls_result, lo_ori_cls_result, lo_ori_reg_result, lo_centroid_result, lo_coeffs_result = posenet(
                    image)
                yaw_cls_loss, yaw_reg_loss = joint_loss(yaw_cls_result, yaw_cls, yaw_reg_result, yaw_reg)
                roll_cls_loss, roll_reg_loss = joint_loss(roll_cls_result, roll_cls, roll_reg_result, roll_reg)
                lo_ori_cls_loss, lo_ori_reg_loss = joint_loss(lo_ori_cls_result, lo_ori_cls, lo_ori_reg_result,
                                                              lo_ori_reg)
                lo_centroid_loss = reg_criterion(lo_centroid_result, lo_centroid) * opt.cls_reg_ratio
                lo_coeffs_loss = reg_criterion(lo_coeffs_result, lo_coeffs) * opt.cls_reg_ratio
                yaw_error = get_yaw_roll_error(bins_tensor['yaw_bin'], yaw_reg_result, yaw_cls_result, yaw_reg, yaw_cls)
                roll_error = get_yaw_roll_error(bins_tensor['roll_bin'], roll_reg_result, roll_cls_result, roll_reg, roll_cls)
                yaw_error_record.update(yaw_error.cpu().numpy(), opt.testBatchSize)
                roll_error_record.update(roll_error.cpu().numpy(), opt.testBatchSize)
                yaw_reg_loss_record.update(yaw_reg_loss.item(), opt.testBatchSize)
                yaw_cls_loss_record.update(yaw_cls_loss.item(), opt.testBatchSize)
                roll_reg_loss_record.update(roll_reg_loss.item(), opt.testBatchSize)
                roll_cls_loss_record.update(roll_cls_loss.item(), opt.testBatchSize)
                lo_ori_cls_loss_record.update(lo_ori_cls_loss.item(), opt.testBatchSize)
                lo_ori_reg_loss_record.update(lo_ori_reg_loss.item(), opt.testBatchSize)
                lo_centroid_loss_record.update(lo_centroid_loss.item(), opt.testBatchSize)
                lo_coeffs_loss_record.update(lo_coeffs_loss.item(), opt.testBatchSize)
                layout_bdb = get_layout_bdb(bins_tensor, torch.argmax(lo_ori_cls_result, 1), lo_ori_reg_result, lo_centroid_result, lo_coeffs_result)
                if not op.exists(op.join(result_save_path, str(sequence_id[0].numpy()))):
                    os.mkdir(op.join(result_save_path, str(sequence_id[0].numpy())))
                savemat(op.join(result_save_path, str(sequence_id[0].numpy()), 'layout.mat'), mdict={'layout': layout_bdb[0, :, :].cpu().numpy()})
            if opt.branch == 'bdbnet' or opt.branch == 'jointnet':
                patch = sequence['boxes_batch']['patch'].to(device)
                bdb2d, bdb3d, bdb_pos, size_reg, size_cls, ori_reg, ori_cls, centroid_reg, centroid_cls, offset_2d = \
                    sequence['boxes_batch']['bdb2d'].float().to(device), \
                    sequence['boxes_batch']['bdb3d'].float().to(device), \
                    sequence['boxes_batch']['bdb_pos'].float().to(device), \
                    sequence['boxes_batch']['size_reg'].float().to(device), \
                    sequence['boxes_batch']['size_cls'].long().to(device), \
                    sequence['boxes_batch']['ori_reg'].float().to(device), \
                    sequence['boxes_batch']['ori_cls'].long().to(device), \
                    sequence['boxes_batch']['centroid_reg'].float().to(device), \
                    sequence['boxes_batch']['centroid_cls'].long().to(device), \
                    sequence['boxes_batch']['delta_2d'].float().to(device)
                size_reg_result, size_cls_result, ori_reg_result, ori_cls_result, centroid_reg_result, centroid_cls_result, offset_2d_result = bdb3dnet(
                    patch)
                size_cls_loss, size_reg_loss = joint_loss(size_cls_result, size_cls, size_reg_result, size_reg)
                ori_cls_loss, ori_reg_loss = joint_loss(ori_cls_result, ori_cls, ori_reg_result, ori_reg)
                centroid_cls_loss, centroid_reg_loss = joint_loss(centroid_cls_result, centroid_cls,
                                                                  centroid_reg_result, centroid_reg)

                #
                offset_2d_loss = reg_criterion(offset_2d_result, offset_2d)
                offset_2d_loss_record.update(offset_2d_loss.item(), opt.testBatchSize)
                size_reg_loss_record.update(size_reg_loss.item(), opt.testBatchSize)
                size_cls_loss_record.update(size_cls_loss.item(), opt.testBatchSize)
                ori_reg_loss_record.update(ori_reg_loss.item(), opt.testBatchSize)
                ori_cls_loss_record.update(ori_cls_loss.item(), opt.testBatchSize)
                centroid_reg_loss_record.update(centroid_reg_loss.item(), opt.testBatchSize)
                centroid_cls_loss_record.update(centroid_cls_loss.item(), opt.testBatchSize)

            if opt.branch == 'jointnet':
                r_ex_result = get_rotation_matix_result(bins_tensor, yaw_cls, yaw_reg_result, roll_cls, roll_reg_result)
                r_ex_gt = get_rotation_matrix_gt(bins_tensor, yaw_cls, yaw_reg, roll_cls, roll_reg)
                P_gt = torch.stack(((bdb_pos[:, 0] + bdb_pos[:, 2]) / 2 - (bdb_pos[:, 2] - bdb_pos[:, 0]) * offset_2d[:, 0], (bdb_pos[:, 1] + bdb_pos[:, 3]) / 2 - (bdb_pos[:, 3] - bdb_pos[:, 1]) * offset_2d[:, 1]), 1)  # P is the center of the bounding boxes
                P_result = torch.stack(((bdb_pos[:, 0] + bdb_pos[:, 2]) / 2 - (bdb_pos[:, 2] - bdb_pos[:, 0]) * offset_2d_result[:, 0], (bdb_pos[:, 1] + bdb_pos[:, 3]) / 2 - (bdb_pos[:, 3] - bdb_pos[:, 1]) * offset_2d_result[:, 1]), 1)  # P is the center of the bounding boxes
                # P is the center of the bounding boxes
                bdb3d_result = get_bdb_3d_result(bins_tensor, ori_cls, ori_reg_result, centroid_cls,
                                                 centroid_reg_result, size_cls, size_reg_result, P_result, K, r_ex_result)
                bdb3d_gt = get_bdb_3d_gt(bins_tensor, ori_cls, ori_reg, centroid_cls, centroid_reg, size_cls, size_reg,
                                         P_gt, K, r_ex_gt)
                corner_loss = 5 * opt.cls_reg_ratio * reg_criterion(bdb3d_result, bdb3d_gt)
                # 2d bdb loss
                bdb2d_result = get_bdb_2d_result(bdb3d_result, r_ex_result, K)
                bdb_loss = 20 * opt.cls_reg_ratio * reg_criterion(bdb2d_result, bdb2d)
                # physical loss
                layout_3d = get_layout_bdb(bins_tensor, lo_ori_cls, lo_ori_reg_result, lo_centroid_result,
                                           lo_coeffs_result)
                phy_violation, phy_gt = physical_violation(layout_3d, bdb3d_result)
                phy_loss = 20 * mse_criterion(phy_violation, phy_gt)
                phy_loss_record.update(phy_loss.item(), opt.testBatchSize)
                r_ex_out = get_rotation_matix_result(bins_tensor, torch.argmax(yaw_cls_result, 1), yaw_reg_result, torch.argmax(roll_cls_result, 1), roll_reg_result)
                predict_bdb = get_bdb_evaluation(bins_tensor, torch.argmax(ori_cls_result, 1), ori_reg_result,
                                            torch.argmax(centroid_cls_result, 1), centroid_reg_result,
                                            size_cls, size_reg_result, P_result, K, r_ex_out)
                # confidence = torch.gather(torch.nn.functional.softmax(size_cls_result), 1, size_cls.view(size_cls.size(0), 1).expand(size_cls.size(0), 1)).squeeze(1)
                for i, evaluate_bdb in enumerate(predict_bdb):
                    class_name = OBJ_CATEGORY_CLEAN[size_cls[i]]
                    if class_name not in IoU3D.keys():
                        IoU3D[class_name] = list()
                        IoU2D[class_name] = list()
                    iou_3d = get_iou_cuboid(get_corners_of_bb3d(evaluate_bdb['basis'], evaluate_bdb['coeffs'], evaluate_bdb['centroid']), bdb3d_gt[i, :, :].cpu().detach().numpy())
                    box_1 = bdb2d_result[i, :].cpu().detach().numpy()
                    box_2 = bdb2d[i, :].cpu().detach().numpy()
                    box_1 = {'x1': box_1[0], 'y1': box_1[1], 'x2': box_1[2], 'y2': box_1[3]}
                    box_2 = {'x1': box_2[0], 'y1': box_2[1], 'x2': box_2[2], 'y2': box_2[3]}
                    iou_2d = get_iou(box_1, box_2)
                    # print sequence_id, class_name, iou_3d, iou_2d
                    IoU3D[class_name].append(iou_3d)
                    IoU2D[class_name].append(iou_2d)

                savemat(op.join(result_save_path, str(sequence_id[0].numpy()), 'bdb_3d.mat'),
                                   mdict={'bdb': predict_bdb})
                savemat(op.join(result_save_path, str(sequence_id[0].numpy()), 'r_ex.mat'), mdict={'r_ex': r_ex_out[0, :, :].cpu().numpy()})
                bdb_loss_record.update(bdb_loss.item(), opt.testBatchSize)
                corner_loss_record.update(corner_loss.item(), opt.testBatchSize)
                total_loss = offset_2d_loss + phy_loss + bdb_loss + size_cls_loss + size_reg_loss + ori_cls_loss + ori_reg_loss + centroid_cls_loss + centroid_reg_loss + corner_loss + opt.obj_cam_ratio * (
                    yaw_cls_loss + yaw_reg_loss + roll_cls_loss + roll_reg_loss + lo_ori_cls_loss + lo_ori_reg_loss + lo_centroid_loss + lo_coeffs_loss)
            if opt.branch == 'posenet':
                total_loss = yaw_cls_loss + yaw_reg_loss + roll_reg_loss + roll_cls_loss + \
                             lo_ori_cls_loss + lo_ori_reg_loss + lo_centroid_loss + lo_coeffs_loss
            if opt.branch == 'bdbnet':
                total_loss = offset_2d_loss + size_cls_loss + size_reg_loss + ori_cls_loss + ori_reg_loss + centroid_cls_loss + centroid_reg_loss + corner_loss
            total_loss_record.update(total_loss.item(), opt.testBatchSize)

    if opt.branch == 'posenet':
        print 'evaluation loss for %d epoch is %f, yaw_cls_loss is %f, yaw_reg_loss is %f, roll_cls_loss is %f, roll_reg_loss is %f, ' \
              'lo_ori_cls_loss is %f, lo_ori_reg_loss is %f, lo_centroid_loss is %f, lo_coeffs_loss is %f' % \
              (epoch, total_loss_record.avg, yaw_cls_loss_record.avg, yaw_reg_loss_record.avg,
               roll_cls_loss_record.avg, roll_reg_loss_record.avg,
               lo_ori_cls_loss_record.avg, lo_ori_reg_loss_record.avg, lo_centroid_loss_record.avg,
               lo_coeffs_loss_record.avg)
    if opt.branch == 'bdbnet':
        print 'evaluation loss for %d epoch is %f, offset_loss is %f, corner_loss is %f, size_cls_loss is %f, size_reg_loss is %f, ori_cls_loss is %f, ori_reg_loss is %f, centroid_cls_loss is %f, centroid_reg_loss is %f' % \
              (epoch, total_loss_record.avg, offset_2d_loss_record.avg, corner_loss_record.avg, size_cls_loss_record.avg,
               size_reg_loss_record.avg, ori_cls_loss_record.avg, ori_reg_loss_record.avg,
               centroid_cls_loss_record.avg,
               centroid_reg_loss_record.avg)
    if opt.branch == 'jointnet':
        print 'evaluation loss for %d epoch is %f, offset_loss is %f, phy_loss is %f, bdb_loss is %f, corner_loss is %f, size_cls_loss is %f, size_reg_loss is %f, ori_cls_loss is %f, ori_reg_loss is %f, ' \
              'centroid_cls_loss is %f, centroid_reg_loss is %f, ' \
              'yaw_cls_loss is %f, yaw_reg_loss is %f, roll_cls_loss is %f, roll_reg_loss is %f, ' \
              'lo_ori_cls_loss is %f, lo_ori_reg_loss is %f, lo_centroid_loss is %f, lo_coeffs_loss is %f' % \
              (epoch, total_loss_record.avg, offset_2d_loss_record.avg, phy_loss_record.avg, bdb_loss_record.avg, corner_loss_record.avg, size_cls_loss_record.avg,
               size_reg_loss_record.avg, ori_cls_loss_record.avg, ori_reg_loss_record.avg,
               centroid_cls_loss_record.avg, centroid_reg_loss_record.avg,
               yaw_cls_loss_record.avg, yaw_reg_loss_record.avg, roll_cls_loss_record.avg, roll_reg_loss_record.avg,
               lo_ori_cls_loss_record.avg, lo_ori_reg_loss_record.avg, lo_centroid_loss_record.avg,
               lo_coeffs_loss_record.avg)
        print 'yaw average error is %f, roll average error is %f' % (yaw_error_record.avg, roll_error_record.avg)
        for key, value in IoU3D.items():
            print '%s: 3d iou is %f, 2d iou is %f' % (key, np.array(IoU3D[key]).mean(), np.array(IoU2D[key]).mean())
    print total_loss_record.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

test_epoch(1)
