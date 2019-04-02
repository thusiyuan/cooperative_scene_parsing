"""
Created on April, 018

@author: Siyuan Huang

Train the SUNCG or SUNRGBD dataset
"""

import torch
import torch.nn as nn
import argparse
import os.path as op
import torch.optim
from models.model_res import Bdb3dNet
from models.model_res import PosNet
from utils.net_utils import get_rotation_matix_result, get_rotation_matrix_gt, to_dict_tensor, get_bdb_2d_result, get_bdb_3d_result, get_bdb_3d_gt, get_layout_bdb, physical_violation
import config


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
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate. Default=0.001')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--metadataPath', type=str, default='metadata/', help='data saving dir')
parser.add_argument('--dataset', type=str, default='sunrgbd', help='sunrgbd or suncg. Default=sunrgbd')
parser.add_argument('--cls_reg_ratio', type=float, default=10, help='the ratio between the loss of classification and regression')
parser.add_argument('--obj_cam_ratio', type=float, default=1, help='the ratio between the loss of classification and regression')
parser.add_argument('--branch', type=str, default='jointnet', help='posenet, bdbnet or jointnet')
parser.add_argument('--rate_decay', type=float, default=10, help='decrease the learning rate by certain epochs')
parser.add_argument('--fine_tune', type=str2bool, default=True, help='whether to fine-tune the model')
parser.add_argument('--pre_train_model_pose', type=str, default='suncg/models_final/posenet_suncg.pth', help='the directory of pre-trained model')
parser.add_argument('--pre_train_model_bdb', type=str, default='suncg/models_final/bdbnet_suncg.pth', help='second model path when train the joint net')


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

pretrained_pose = op.join(opt.metadataPath, opt.pre_train_model_pose)
pretrained_bdb = op.join(opt.metadataPath, opt.pre_train_model_bdb)

if opt.fine_tune:
    if opt.branch == 'posenet':
        posenet.load_weight(pretrained_pose)
        posenet.freeze_res_layer(7)
    if opt.branch == 'bdbnet':
        bdb3dnet.load_weight(pretrained_bdb)
        bdb3dnet.freeze_res_layer(5)
    if opt.branch == 'jointnet':
        posenet.load_weight(pretrained_pose)
        posenet.freeze_res_layer(6)
        bdb3dnet.load_weight(pretrained_bdb)
        bdb3dnet.freeze_res_layer(5)
elif opt.dataset == 'sunrgbd':  # if we directly train on SUNRGBD, we fix several modules since SUNRGBD is relative small
    if opt.branch == 'posenet' or opt.branch == 'jointnet':
        posenet.freeze_res_layer(6)
    if opt.branch == 'bdbnet' or opt.branch == 'jointnet':
        bdb3dnet.freeze_res_layer(6)

if opt.dataset == 'sunrgbd':
    from data.sunrgbd import sunrgbd_train_loader, sunrgbd_test_loader
    train_loader = sunrgbd_train_loader(opt)
    test_loader = sunrgbd_test_loader(opt)
if opt.dataset == 'suncg':
    from data.suncg import suncg_train_loader, suncg_test_loader
    train_loader = suncg_train_loader(opt)
    test_loader = suncg_test_loader(opt)

if opt.branch == 'posenet':
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, posenet.parameters()), lr=opt.lr)
elif opt.branch == 'bdbnet':
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, bdb3dnet.parameters()), lr=opt.lr)
elif opt.branch == 'jointnet':
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, posenet.parameters()) + filter(lambda p: p.requires_grad, bdb3dnet.parameters()), lr=opt.lr)


cls_criterion = nn.CrossEntropyLoss(size_average=True, reduce=True)
reg_criterion = nn.SmoothL1Loss(size_average=True, reduce=True)
mse_criterion = nn.MSELoss(size_average=True, reduce=True)


def joint_loss(cls_result, cls_gt, reg_result, reg_gt):
    cls_loss = cls_criterion(cls_result, cls_gt)
    if len(reg_result.size()) == 3:
        reg_result = torch.gather(reg_result, 1, cls_gt.view(reg_gt.size(0), 1, 1).expand(reg_gt.size(0), 1, reg_gt.size(1)))
    else:
        reg_result = torch.gather(reg_result, 1, cls_gt.view(reg_gt.size(0), 1).expand(reg_gt.size(0), 1))
    reg_result = reg_result.squeeze(1)
    reg_loss = reg_criterion(reg_result, reg_gt)
    return cls_loss, opt.cls_reg_ratio * reg_loss


def train_epoch(epoch):
    # initial recorder
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
    offset_2d_loss_record = AverageMeter()
    if opt.branch == 'posenet':
        posenet.train()
    if opt.branch == 'bdbnet':
        bdb3dnet.train()
    if opt.branch == 'jointnet':    # Batch size is 1 during the training of jointnet, so we freeze the BN layers during training
        posenet.train()
        bdb3dnet.train()
        posenet.freeze_bn_layer()
        bdb3dnet.freeze_bn_layer()
    # load data
    for i, sequence in enumerate(train_loader):
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
            yaw_reg_result, roll_reg_result, yaw_cls_result, roll_cls_result, lo_ori_cls_result, lo_ori_reg_result, lo_centroid_result, lo_coeffs_result = posenet(image)
            yaw_cls_loss, yaw_reg_loss = joint_loss(yaw_cls_result, yaw_cls, yaw_reg_result, yaw_reg)
            roll_cls_loss, roll_reg_loss = joint_loss(roll_cls_result, roll_cls, roll_reg_result, roll_reg)
            lo_ori_cls_loss, lo_ori_reg_loss = joint_loss(lo_ori_cls_result, lo_ori_cls, lo_ori_reg_result, lo_ori_reg)
            lo_centroid_loss = reg_criterion(lo_centroid_result, lo_centroid) * opt.cls_reg_ratio
            lo_coeffs_loss = reg_criterion(lo_coeffs_result, lo_coeffs) * opt.cls_reg_ratio
            yaw_reg_loss_record.update(yaw_reg_loss.item(), opt.batchSize)
            yaw_cls_loss_record.update(yaw_cls_loss.item(), opt.batchSize)
            roll_reg_loss_record.update(roll_reg_loss.item(), opt.batchSize)
            roll_cls_loss_record.update(roll_cls_loss.item(), opt.batchSize)
            lo_ori_cls_loss_record.update(lo_ori_cls_loss.item(), opt.batchSize)
            lo_ori_reg_loss_record.update(lo_ori_reg_loss.item(), opt.batchSize)
            lo_centroid_loss_record.update(lo_centroid_loss.item(), opt.batchSize)
            lo_coeffs_loss_record.update(lo_coeffs_loss.item(), opt.batchSize)
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
            size_reg_result, size_cls_result, ori_reg_result, ori_cls_result, centroid_reg_result, centroid_cls_result, offset_2d_result = bdb3dnet(patch)
            size_cls_loss, size_reg_loss = joint_loss(size_cls_result, size_cls, size_reg_result, size_reg)
            ori_cls_loss, ori_reg_loss = joint_loss(ori_cls_result, ori_cls, ori_reg_result, ori_reg)
            centroid_cls_loss, centroid_reg_loss = joint_loss(centroid_cls_result, centroid_cls, centroid_reg_result, centroid_reg)
            #
            offset_2d_loss = reg_criterion(offset_2d_result, offset_2d)
            size_reg_loss_record.update(size_reg_loss.item(), opt.batchSize)
            size_cls_loss_record.update(size_cls_loss.item(), opt.batchSize)
            ori_reg_loss_record.update(ori_reg_loss.item(), opt.batchSize)
            ori_cls_loss_record.update(ori_cls_loss.item(), opt.batchSize)
            centroid_reg_loss_record.update(centroid_reg_loss.item(), opt.batchSize)
            centroid_cls_loss_record.update(centroid_cls_loss.item(), opt.batchSize)
            offset_2d_loss_record.update(offset_2d_loss.item(), opt.batchSize)
        if opt.branch == 'jointnet':
            # 3d bdb loss
            r_ex_result = get_rotation_matix_result(bins_tensor, yaw_cls, yaw_reg_result, roll_cls, roll_reg_result)
            r_ex_gt = get_rotation_matrix_gt(bins_tensor, yaw_cls, yaw_reg, roll_cls, roll_reg)
            # apply the 2D offset to the bounding box center
            P_gt = torch.stack(((bdb_pos[:, 0] + bdb_pos[:, 2]) / 2 - (bdb_pos[:, 2] - bdb_pos[:, 0]) * offset_2d[:, 0],
                                (bdb_pos[:, 1] + bdb_pos[:, 3]) / 2 - (bdb_pos[:, 3] - bdb_pos[:, 1]) * offset_2d[:,
                                                                                                        1]),
                               1)  # P is the center of the bounding boxes
            P_result = torch.stack(((bdb_pos[:, 0] + bdb_pos[:, 2]) / 2 - (
            bdb_pos[:, 2] - bdb_pos[:, 0]) * offset_2d_result[:, 0], (bdb_pos[:, 1] + bdb_pos[:, 3]) / 2 - (
                                    bdb_pos[:, 3] - bdb_pos[:, 1]) * offset_2d_result[:, 1]),
                                   1)  # P is the center of the bounding boxes
            bdb3d_result = get_bdb_3d_result(bins_tensor, ori_cls, ori_reg_result, centroid_cls, centroid_reg_result, size_cls, size_reg_result, P_result, K, r_ex_result)
            bdb3d_gt = get_bdb_3d_gt(bins_tensor, ori_cls, ori_reg, centroid_cls, centroid_reg, size_cls, size_reg, P_gt, K, r_ex_gt)
            corner_loss = 5 * opt.cls_reg_ratio * reg_criterion(bdb3d_result, bdb3d_gt)
            # 2d bdb loss
            bdb2d_result = get_bdb_2d_result(bdb3d_result, r_ex_result, K)
            bdb_loss = 20 * opt.cls_reg_ratio * reg_criterion(bdb2d_result, bdb2d)
            # physical loss
            layout_3d = get_layout_bdb(bins_tensor, lo_ori_cls, lo_ori_reg_result, lo_centroid_result, lo_coeffs_result)
            phy_violation, phy_gt = physical_violation(layout_3d, bdb3d_result)
            phy_loss = 20 * mse_criterion(phy_violation, phy_gt)
            phy_loss_record.update(phy_loss.item(), opt.batchSize)
            bdb_loss_record.update(bdb_loss.item(), opt.batchSize)
            corner_loss_record.update(corner_loss.item(), opt.batchSize)
            total_loss = offset_2d_loss + phy_loss + bdb_loss + size_cls_loss + size_reg_loss + ori_cls_loss + ori_reg_loss + centroid_cls_loss + centroid_reg_loss + corner_loss + opt.obj_cam_ratio * (
                yaw_cls_loss + yaw_reg_loss + roll_cls_loss + roll_reg_loss + lo_ori_cls_loss + lo_ori_reg_loss + lo_centroid_loss + lo_coeffs_loss)
        if opt.branch == 'posenet':
            total_loss = yaw_cls_loss + yaw_reg_loss + roll_reg_loss + roll_cls_loss + \
                         lo_ori_cls_loss + lo_ori_reg_loss + lo_centroid_loss + lo_coeffs_loss
        if opt.branch == 'bdbnet':
            total_loss = offset_2d_loss + size_cls_loss + size_reg_loss + ori_cls_loss + ori_reg_loss + centroid_cls_loss + centroid_reg_loss
        total_loss_record.update(total_loss.item(), opt.batchSize)

        if (i + 1) % 25000 == 0:
            print epoch, i + 1, total_loss.item()
            if opt.branch == 'posenet':
                print yaw_cls_loss.item(), yaw_reg_loss.item(), roll_cls_loss.item(), roll_reg_loss.item(), \
                    lo_ori_cls_loss.item(), lo_ori_reg_loss.item(), lo_centroid_loss.item(), lo_coeffs_loss.item()
            if opt.branch == 'bdbnet':
                print 'corner_loss is %f, size_cls_loss is %f, size_reg_loss is %f, ori_cls_loss is %f, ori_reg_loss is %f, centroid_cls_loss is %f, centroid_reg_loss is %f' % \
               (corner_loss.item(), size_cls_loss.item(), size_reg_loss.item(), ori_cls_loss.item(), ori_reg_loss.item(), centroid_cls_loss.item(),
               centroid_reg_loss.item())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    if opt.branch == 'posenet':
        print 'training loss for %d epoch is %f, yaw_cls_loss is %f, yaw_reg_loss is %f, roll_cls_loss is %f, roll_reg_loss is %f, ' \
              'lo_ori_cls_loss is %f, lo_ori_reg_loss is %f, lo_centroid_loss is %f, lo_coeffs_loss is %f' % \
              (epoch, total_loss_record.avg, yaw_cls_loss_record.avg, yaw_reg_loss_record.avg, roll_cls_loss_record.avg, roll_reg_loss_record.avg,
               lo_ori_cls_loss_record.avg, lo_ori_reg_loss_record.avg, lo_centroid_loss_record.avg, lo_coeffs_loss_record.avg)
    if opt.branch == 'bdbnet':
        print 'training loss for %d epoch is %f, offset_loss is %f, size_cls_loss is %f, size_reg_loss is %f, ori_cls_loss is %f, ori_reg_loss is %f, centroid_cls_loss is %f, centroid_reg_loss is %f' % \
              (epoch, total_loss_record.avg, offset_2d_loss_record.avg, size_cls_loss_record.avg, size_reg_loss_record.avg, ori_cls_loss_record.avg, ori_reg_loss_record.avg, centroid_cls_loss_record.avg,
               centroid_reg_loss_record.avg)
    if opt.branch == 'jointnet':
        print 'training loss for %d epoch is %f, offset_loss is %f, physical loss is %f, bdb_loss is %f, corner_loss is %f, size_cls_loss is %f, size_reg_loss is %f, ori_cls_loss is %f, ori_reg_loss is %f, ' \
              'centroid_cls_loss is %f, centroid_reg_loss is %f, ' \
              'yaw_cls_loss is %f, yaw_reg_loss is %f, roll_cls_loss is %f, roll_reg_loss is %f, ' \
              'lo_ori_cls_loss is %f, lo_ori_reg_loss is %f, lo_centroid_loss is %f, lo_coeffs_loss is %f' % \
              (epoch, total_loss_record.avg, offset_2d_loss_record.avg, phy_loss_record.avg, bdb_loss_record.avg, corner_loss_record.avg, size_cls_loss_record.avg, size_reg_loss_record.avg, ori_cls_loss_record.avg, ori_reg_loss_record.avg,
               centroid_cls_loss_record.avg, centroid_reg_loss_record.avg,
               yaw_cls_loss_record.avg, yaw_reg_loss_record.avg, roll_cls_loss_record.avg, roll_reg_loss_record.avg,
               lo_ori_cls_loss_record.avg, lo_ori_reg_loss_record.avg, lo_centroid_loss_record.avg, lo_coeffs_loss_record.avg)


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
    offset_2d_loss_record = AverageMeter()
    if opt.branch == 'posenet' or opt.branch == 'jointnet':
        posenet.eval()
    if opt.branch == 'bdbnet' or opt.branch == 'jointnet':
        bdb3dnet.eval()
    with torch.no_grad():
        for i, sequence in enumerate(test_loader):
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
                yaw_reg_result, roll_reg_result, yaw_cls_result, roll_cls_result, lo_ori_cls_result, lo_ori_reg_result, lo_centroid_result, lo_coeffs_result = posenet(
                    image)
                yaw_cls_loss, yaw_reg_loss = joint_loss(yaw_cls_result, yaw_cls, yaw_reg_result, yaw_reg)
                roll_cls_loss, roll_reg_loss = joint_loss(roll_cls_result, roll_cls, roll_reg_result, roll_reg)
                lo_ori_cls_loss, lo_ori_reg_loss = joint_loss(lo_ori_cls_result, lo_ori_cls, lo_ori_reg_result,
                                                              lo_ori_reg)
                lo_centroid_loss = reg_criterion(lo_centroid_result, lo_centroid) * opt.cls_reg_ratio
                lo_coeffs_loss = reg_criterion(lo_coeffs_result, lo_coeffs) * opt.cls_reg_ratio
                yaw_reg_loss_record.update(yaw_reg_loss.item(), opt.batchSize)
                yaw_cls_loss_record.update(yaw_cls_loss.item(), opt.batchSize)
                roll_reg_loss_record.update(roll_reg_loss.item(), opt.batchSize)
                roll_cls_loss_record.update(roll_cls_loss.item(), opt.batchSize)
                lo_ori_cls_loss_record.update(lo_ori_cls_loss.item(), opt.batchSize)
                lo_ori_reg_loss_record.update(lo_ori_reg_loss.item(), opt.batchSize)
                lo_centroid_loss_record.update(lo_centroid_loss.item(), opt.batchSize)
                lo_coeffs_loss_record.update(lo_coeffs_loss.item(), opt.batchSize)
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
                size_reg_loss_record.update(size_reg_loss.item(), opt.batchSize)
                size_cls_loss_record.update(size_cls_loss.item(), opt.batchSize)
                ori_reg_loss_record.update(ori_reg_loss.item(), opt.batchSize)
                ori_cls_loss_record.update(ori_cls_loss.item(), opt.batchSize)
                centroid_reg_loss_record.update(centroid_reg_loss.item(), opt.batchSize)
                centroid_cls_loss_record.update(centroid_cls_loss.item(), opt.batchSize)
                offset_2d_loss_record.update(offset_2d_loss.item(), opt.batchSize)

            if opt.branch == 'jointnet':
                # 3d bdb loss
                r_ex_result = get_rotation_matix_result(bins_tensor, yaw_cls, yaw_reg_result, roll_cls, roll_reg_result)
                r_ex_gt = get_rotation_matrix_gt(bins_tensor, yaw_cls, yaw_reg, roll_cls, roll_reg)
                P_gt = torch.stack(
                    ((bdb_pos[:, 0] + bdb_pos[:, 2]) / 2 - (bdb_pos[:, 2] - bdb_pos[:, 0]) * offset_2d[:, 0],
                     (bdb_pos[:, 1] + bdb_pos[:, 3]) / 2 - (bdb_pos[:, 3] - bdb_pos[:, 1]) * offset_2d[:,
                                                                                             1]),
                    1)  # P is the center of the bounding boxes
                P_result = torch.stack(((bdb_pos[:, 0] + bdb_pos[:, 2]) / 2 - (
                    bdb_pos[:, 2] - bdb_pos[:, 0]) * offset_2d_result[:, 0], (bdb_pos[:, 1] + bdb_pos[:, 3]) / 2 - (
                                            bdb_pos[:, 3] - bdb_pos[:, 1]) * offset_2d_result[:, 1]),
                                       1)  # P is the center of the bounding boxes
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
                phy_loss_record.update(phy_loss.item(), opt.batchSize)
                bdb_loss_record.update(bdb_loss.item(), opt.batchSize)
                corner_loss_record.update(corner_loss.item(), opt.batchSize)
                total_loss = offset_2d_loss + phy_loss + bdb_loss + size_cls_loss + size_reg_loss + ori_cls_loss + ori_reg_loss + centroid_cls_loss + centroid_reg_loss + corner_loss + opt.obj_cam_ratio * (
                    yaw_cls_loss + yaw_reg_loss + roll_cls_loss + roll_reg_loss + lo_ori_cls_loss + lo_ori_reg_loss + lo_centroid_loss + lo_coeffs_loss)
            if opt.branch == 'posenet':
                total_loss = yaw_cls_loss + yaw_reg_loss + roll_reg_loss + roll_cls_loss + \
                             lo_ori_cls_loss + lo_ori_reg_loss + lo_centroid_loss + lo_coeffs_loss
            if opt.branch == 'bdbnet':
                total_loss = offset_2d_loss + size_cls_loss + size_reg_loss + ori_cls_loss + ori_reg_loss + centroid_cls_loss + centroid_reg_loss
            total_loss_record.update(total_loss.item(), opt.batchSize)

        if opt.branch == 'posenet':
            print 'evaluation loss for %d epoch is %f, yaw_cls_loss is %f, yaw_reg_loss is %f, roll_cls_loss is %f, roll_reg_loss is %f, ' \
                  'lo_ori_cls_loss is %f, lo_ori_reg_loss is %f, lo_centroid_loss is %f, lo_coeffs_loss is %f' % \
                  (epoch, total_loss_record.avg, yaw_cls_loss_record.avg, yaw_reg_loss_record.avg,
                   roll_cls_loss_record.avg, roll_reg_loss_record.avg,
                   lo_ori_cls_loss_record.avg, lo_ori_reg_loss_record.avg, lo_centroid_loss_record.avg,
                   lo_coeffs_loss_record.avg)
        if opt.branch == 'bdbnet':
            print 'evaluation loss for %d epoch is %f, offset_loss is %f, size_cls_loss is %f, size_reg_loss is %f, ori_cls_loss is %f, ori_reg_loss is %f, centroid_cls_loss is %f, centroid_reg_loss is %f' % \
                  (epoch, total_loss_record.avg, offset_2d_loss_record.avg, size_cls_loss_record.avg,
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
    return total_loss_record.avg


def train():
    min_eval_loss = 0
    for epoch in range(1, opt.nEpochs + 1):
        adjust_learning_rate(optimizer, epoch)
        train_epoch(epoch)
        total_loss = test_epoch(epoch)
        if min_eval_loss == 0 or total_loss < min_eval_loss:
            checkpoint(epoch)
            min_eval_loss = total_loss


def checkpoint(epoch):
    if opt.branch == 'posenet':
        model_out_path = op.join(opt.metadataPath, opt.dataset, 'models', 'posenet_' + str(epoch) + '.pth')
        torch.save(posenet, model_out_path)
    if opt.branch == 'bdbnet':
        model_out_path = op.join(opt.metadataPath, opt.dataset, 'models', 'bdbnet_' + str(epoch)
                                 + '.pth')
        torch.save(bdb3dnet, model_out_path)
    if opt.branch == 'jointnet':
        pose_out_path = op.join(opt.metadataPath, opt.dataset, 'models', 'joint_posenet_' + str(epoch) + '.pth')
        torch.save(posenet, pose_out_path)
        bdb_out_path = op.join(opt.metadataPath, opt.dataset, 'models', 'joint_bdbnet_' + str(epoch) + '.pth')
        torch.save(bdb3dnet, bdb_out_path)


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


def adjust_learning_rate(optimizer, epoch):
    lr = opt.lr * (0.7 ** (epoch // opt.rate_decay))
    lr = max(lr, 0.00001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    train()


if __name__ == "__main__":
    main()