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
import time


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
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--metadataPath', type=str, default='/home/siyuan/Documents/nips2018', help='data saving dir')
parser.add_argument('--dataset', type=str, default='sunrgbd', help='sunrgbd or suncg. Default=sunrgbd')
parser.add_argument('--cls_reg_ratio', type=float, default=10, help='the ratio between the loss of classification and regression')
parser.add_argument('--obj_cam_ratio', type=float, default=2, help='the ratio between the loss of classification and regression')
parser.add_argument('--branch', type=str, default='jointnet', help='posenet, bdbnet or jointnet')
parser.add_argument('--model_path', type=str, default='sunrgbd/models_final/joint_posenet_full.pth', help='the directory of trained model')
parser.add_argument('--model_path_2', type=str, default='sunrgbd/models_final/joint_bdbnet_full.pth')
parser.add_argument('--vis', type=bool, default=False, help='whether to visualize the result')
parser.add_argument('--save_result', type=bool, default=False, help='whether to save the result')
parser.add_argument('--inference_file', type=str, default='/home/siyuan/Documents/Deformable-ConvNets/rfcn/test/inference_set.json', help='inference set')
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

pretrained_path = op.join(opt.metadataPath, opt.model_path)
pretrained_path_2 = op.join(opt.metadataPath, opt.model_path_2)
if opt.branch == 'posenet':
    posenet.load_weight(pretrained_path)
if opt.branch == 'bdbnet':
    bdb3dnet.load_weight(pretrained_path_2)
if opt.branch == 'jointnet':
    posenet.load_weight(pretrained_path)
    bdb3dnet.load_weight(pretrained_path_2)

from data.sunrgbd import inference_data_loader
test_loader = inference_data_loader(opt)

cls_criterion = nn.CrossEntropyLoss(size_average=True, reduce=True)
reg_criterion = nn.SmoothL1Loss(size_average=True, reduce=True)
mse_criterion = nn.MSELoss(size_average=True, reduce=True)

result_save_path = op.join(opt.metadataPath, opt.dataset, 'results_test')
if not op.exists(result_save_path):
    os.mkdir(result_save_path)


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


def inference():
    if opt.branch == 'posenet' or opt.branch == 'jointnet':
        posenet.eval()
    if opt.branch == 'bdbnet' or opt.branch == 'jointnet':
        bdb3dnet.eval()
    with torch.no_grad():
        for i, sequence in enumerate(test_loader):
            sequence_id = sequence['sequence_id']
            if opt.branch == 'posenet' or opt.branch == 'jointnet':
                image = sequence['image'].to(device)
                K = sequence['camera']['K'].float().to(device)
                size_cls = sequence['boxes_batch']['size_cls'].long().to(device)
                bdb_pos = sequence['boxes_batch']['bdb_pos'].float().to(device)
                yaw_reg_result, roll_reg_result, yaw_cls_result, roll_cls_result, lo_ori_cls_result, lo_ori_reg_result, lo_centroid_result, lo_coeffs_result = posenet(
                    image)
                layout_bdb = get_layout_bdb(bins_tensor,     torch.argmax(lo_ori_cls_result, 1), lo_ori_reg_result, lo_centroid_result, lo_coeffs_result)
                if not op.exists(op.join(result_save_path, str(sequence_id[0].numpy()))):
                    os.mkdir(op.join(result_save_path, str(sequence_id[0].numpy())))
                savemat(op.join(result_save_path, str(sequence_id[0].numpy()), 'layout.mat'), mdict={'layout': layout_bdb[0, :, :].cpu().numpy()})
            if opt.branch == 'bdbnet' or opt.branch == 'jointnet':
                patch = sequence['boxes_batch']['patch'].to(device)
                size_reg_result, size_cls_result, ori_reg_result, ori_cls_result, centroid_reg_result, centroid_cls_result, offset_2d_result = bdb3dnet(
                    patch)

            if opt.branch == 'jointnet':
                P_result = torch.stack(((bdb_pos[:, 0] + bdb_pos[:, 2]) / 2 + (bdb_pos[:, 2] - bdb_pos[:, 0]) * offset_2d_result[:, 0], (bdb_pos[:, 1] + bdb_pos[:, 3]) / 2 + (bdb_pos[:, 3] - bdb_pos[:, 1]) * offset_2d_result[:, 1]), 1)  # P is the center of the bounding boxes
                # P is the center of the bounding boxes

                r_ex_out = get_rotation_matix_result(bins_tensor, torch.argmax(yaw_cls_result, 1), yaw_reg_result, torch.argmax(roll_cls_result, 1), roll_reg_result)
                predict_bdb = get_bdb_evaluation(bins_tensor, torch.argmax(ori_cls_result, 1), ori_reg_result,
                                            torch.argmax(centroid_cls_result, 1), centroid_reg_result,
                                            size_cls, size_reg_result, P_result, K, r_ex_out)
                savemat(op.join(result_save_path, str(sequence_id[0].numpy()), 'bdb_3d.mat'),
                                   mdict={'bdb': predict_bdb})
                savemat(op.join(result_save_path, str(sequence_id[0].numpy()), 'r_ex.mat'), mdict={'r_ex': r_ex_out[0, :, :].cpu().numpy()})



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


inference()
