%% show our result
clc;
clear all;
addpath(genpath('.'))
load('SUNRGBDMetaUS.mat')
% load('./Metadata/SUNRGBDMeta3DBB_v2.mat');
result_path = '/home/siyuan/Documents/nips2018/sunrgbd/results';
dgp_path = '/home/siyuan/Documents/indoorunderstanding_3dgp/3dgp_evaluation';
% load('./Metadata/SUNRGBD2Dseg.mat')
%% change path
imageId = 103;
% for i = 1:size(SUNRGBDMeta, 2)
%     SUNRGBDMeta(i).depthpath = strrep(SUNRGBDMeta(i).depthpath, '/n/fs/sun3d/data/', '/home/ruiqigao/siyuan/experiments/');
%     SUNRGBDMeta(i).rgbpath = strrep(SUNRGBDMeta(i).rgbpath, '/n/fs/sun3d/data/', '/home/ruiqigao/siyuan/experiments/'); 
% end
% save('SUNRGBDMetaS.mat', 'SUNRGBDMeta');
data = SUNRGBDMeta(imageId);
[rgb,points3d,depthInpaint,imsize]=read3dPoints(data);


%%
layout_path = fullfile(result_path, int2str(imageId), 'layout.mat');
dgp_path = fullfile(dgp_path, int2str(imageId), 'layout.mat');
roomLayout = load(layout_path);
roomLayout_dgp = load(dgp_path);

%% draw 3D 
figure(2);
vis_point_cloud(points3d,rgb, 10000)
hold on;
Linewidth = 6;
maxhight = 1.2;
drawRoom(roomLayout.layout',color_trans('#C44D30'),Linewidth,maxhight);
drawRoom(roomLayout_dgp.layout',color_trans('#D6B0AB'),Linewidth,maxhight);
hold off;
set(gca,'xtick',[]);
set(gca,'ytick',[]);
set(gca,'ztick',[]);
axis off;
%%

