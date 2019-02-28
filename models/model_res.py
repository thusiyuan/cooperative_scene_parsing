"""
Created on April, 2018

@author: Siyuan Huang

Preprocess the SUNRGBD dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

OBJ_SIZE_BIN = 16
OBJ_ORI_BIN = 6
OBJ_CENTER_BIN = 6
YAW_ROLL_BIN = 2
NUM_CLASS = 30
LAYOUT_ORI_BIN = 3


class PosNet(nn.Module):
    def __init__(self):
        super(PosNet, self).__init__()
        self.resnet = resnet.resnet34(pretrained=False)
        self.fc_1 = nn.Linear(2048, 1024)
        self.fc_2 = nn.Linear(1024, YAW_ROLL_BIN * 4)
        # fc for layout
        self.fc_layout = nn.Linear(2048, 2048)
        # fc for layout orientation
        self.fc_3 = nn.Linear(2048, 1024)
        self.fc_4 = nn.Linear(1024, LAYOUT_ORI_BIN * 2)
        # fc for layout centroid
        self.fc_5 = nn.Linear(2048, 1024)
        self.fc_6 = nn.Linear(1024, 6)
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, nn.BatchNorm2d):
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = self.resnet.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # for _, v in pretrained_dict.items():
        #     v.requires_grad = False
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)

    def load_weight(self, model_path):
        model_dict = self.state_dict()
        pretrained_model = torch.load(model_path).state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def freeze_res_layer(self, layer_num=9):
        child_counter = 0
        for child in self.resnet.children():
            if child_counter < layer_num:
                for param in child.parameters():
                    param.requires_grad = False
            child_counter += 1

    def freeze_bn_layer(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        x = self.resnet(x)

        # branch for camera parameter
        cam = self.fc_1(x)
        cam = F.leaky_relu(cam, 0.2)
        cam = F.dropout(cam, 0.5, training=True)
        cam = self.fc_2(cam)
        angle = cam.view(-1, YAW_ROLL_BIN, 4)
        yaw_reg = angle[:, :, 0]
        roll_reg = angle[:, :, 1]
        yaw_cla = angle[:, :, 2]
        roll_cla = angle[:, :, 3]

        # branch for layout
        lo = self.fc_layout(x)
        lo = F.leaky_relu(lo, 0.2)
        lo = F.dropout(lo, 0.5, training=True)
        # branch for layout orientation
        lo_ori = self.fc_3(lo)
        lo_ori = F.leaky_relu(lo_ori, 0.2)
        lo_ori = F.dropout(lo_ori, 0.5, training=True)
        lo_ori = self.fc_4(lo_ori)
        lo_ori = lo_ori.view(-1, LAYOUT_ORI_BIN, 2)
        lo_ori_reg = lo_ori[:, :, 0]
        lo_ori_cla = lo_ori[:, :, 1]
        # branch for layout centroid and coeffs
        lo_ct = self.fc_5(lo)
        lo_ct = F.leaky_relu(lo_ct, 0.2)
        lo_ct = F.dropout(lo_ct, 0.5, training=True)
        lo_ct = self.fc_6(lo_ct)
        lo_ct = lo_ct.view(-1, 6)
        lo_centroid = lo_ct[:, :3]
        lo_coeffs = lo_ct[:, 3:]
        return yaw_reg, roll_reg, yaw_cla, roll_cla, lo_ori_cla, lo_ori_reg, lo_centroid, lo_coeffs


class Bdb3dNet(nn.Module):
    def __init__(self):
        super(Bdb3dNet, self).__init__()
        # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
        self.resnet = resnet.resnet34(pretrained=False)
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, NUM_CLASS * 4)
        self.fc3 = nn.Linear(2048, 256)
        self.fc4 = nn.Linear(256, OBJ_ORI_BIN * 2)
        self.fc5 = nn.Linear(2048, 256)
        self.fc_centroid = nn.Linear(256, OBJ_CENTER_BIN * 2)
        self.fc_off_1 = nn.Linear(2048, 256)
        self.fc_off_2 = nn.Linear(256, 2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = self.resnet.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # for _, v in pretrained_dict.items():
        #     v.requires_grad = False
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)

    def load_weight(self, model_path):
        model_dict = self.state_dict()
        pretrained_model = torch.load(model_path).state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def freeze_res_layer(self, layer_num=9):
        child_counter = 0
        for child in self.resnet.children():
            if child_counter < layer_num:
                for param in child.parameters():
                    param.requires_grad = False
            child_counter += 1

    def freeze_bn_layer(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        x = self.resnet(x)
        flatten = x.view(x.size(0), -1)
        # branch to predict the size
        size = self.fc1(flatten)
        size = F.leaky_relu(size, 0.2)
        size = F.dropout(size, 0.5, training=True)
        size = self.fc2(size)
        size = size.view(-1, NUM_CLASS, 4)
        size_reg = size[:, :, :3]
        size_cls = size[:, :, 3]

        # branch to predict the orientation
        ori = self.fc3(flatten)
        ori = F.leaky_relu(ori, 0.2)
        ori = F.dropout(ori, 0.5, training=True)
        ori = self.fc4(ori)
        ori = ori.view(-1, OBJ_ORI_BIN, 2)
        ori_reg = ori[:, :, 0]
        ori_cls = ori[:, :, 1]

        # branch to predict the centroid
        centroid = self.fc5(flatten)
        centroid = F.leaky_relu(centroid, 0.2)
        centroid = F.dropout(centroid, 0.5, training=True)
        centroid = self.fc_centroid(centroid)
        centroid = centroid.view(-1, OBJ_CENTER_BIN, 2)
        centroid_cls = centroid[:, :, 0]
        centroid_reg = centroid[:, :, 1]

        # branch to predict the 2D offset
        offset = self.fc_off_1(flatten)
        offset = F.leaky_relu(offset, 0.2)
        offset = F.dropout(offset, 0.5, training=True)
        offset = self.fc_off_2(offset)
        return size_reg, size_cls, ori_reg, ori_cls, centroid_reg, centroid_cls, offset

# bdnet = Bdb3dNet()
# posnet = PosNet()
# input_1 = Variable(torch.randn(3, 3, 256, 256))
# input_2 = Variable(torch.randn(3, NUM_CLASS))
# # output = bdnet(input_1)
# output = posnet(input_1)
# output1 = bdnet(input_1)
# print output[0].data.numpy()
# print torch.max(output[2].data, 1)
# print output[0].size(), output[1].size(), output[2].size(), output[3].size(), output[4].size(), output[5].size(), output[6].size(), output[7].size()
# print output1[4].size(), output1[5].size()