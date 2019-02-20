%% show our result
clc;
close all;
clear all;
addpath(genpath('.'))
load('SUNRGBDMetaUS.mat')
result_path = '/home/siyuan/Documents/cvpr2019_scene_graph/sunrgbd/results_full';
%% change path
for imageId = 5 %480
disp(i);
bdb_3d = {};
bdb_2d = {};
disp(imageId);
bdb_result_path = fullfile(result_path, int2str(imageId), 'bdb_3d.mat');
adjacency_matrix_path = fullfile(result_path, int2str(imageId), 'adjacency.mat');
layout_path = fullfile(result_path, int2str(imageId), 'layout.mat');
if ~exist(bdb_result_path, 'file')
    continue;
end
fig = figure;
r_path = fullfile(result_path, int2str(imageId), 'r_ex.mat');
load(r_path);
load(bdb_result_path);
load(adjacency_matrix_path);
adjacency
data = SUNRGBDMeta(imageId);
[rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
%%
for i = 1:size(bdb, 2)
    bdb_3d{i} = bdb{1, i};
end
roomLayout = load(layout_path);

% get 2d corners
for kk = 1:length(bdb_3d)
    bdb_2d_temp = get_corners_of_bb3d(bdb_3d{kk});
    bdb_2d_temp = bdb_2d_temp(:, [1, 3, 2]);
    bdb_2d_temp(:, 2) = - bdb_2d_temp(:, 2);
    bdb_2d_temp = (data.K*r_ex*bdb_2d_temp')';
    bdb_2d_corner(:, 1) = bdb_2d_temp(:, 1) ./ bdb_2d_temp(:, 3);
    bdb_2d_corner(:, 2) = bdb_2d_temp(:, 2) ./ bdb_2d_temp(:, 3);
    bdb_2d{kk} = bdb_2d_corner;
end

%% draw 2D
imshow(data.rgbpath);
hold on; 
for kk =1:length(bdb_2d)
    draw_square_3d(bdb_2d{kk}, myObjectColor(kk), 5);
    x_max = min(bdb_2d{kk});
    text(double(x_max(1)), double(x_max(2)), num2str(kk), 'Color','red','FontSize',14);
end
hold off;
set(gca,'xtick',[])
set(gca,'ytick',[])
axis off;
% clf;
% print(fig, ['results/'  num2str(imageId) '_2d'], '-dpng')
% saveas(fig, ['results/'  num2str(imageId) '_2d.png']);
% close(fig);
pause(1);
%% draw 3D 
fig = figure;
vis_point_cloud(points3d,rgb, 100000)
hold on;
for kk = 1:length(bdb_3d)
    vis_cube(bdb_3d{kk}, myObjectColor(kk), 2);
%     vis_cube(data.groundtruth3DBB(kk), myObjectColor(kk), 2);
end
Linewidth = 3;
maxhight = 1.2;
drawRoom(roomLayout.layout','b',Linewidth,maxhight);
hold off;
set(gca,'xtick',[])
set(gca,'ytick',[])
set(gca, 'ztick', [])
axis off;
% print(fig, ['results/'  num2str(imageId) '_3d'], '-dpng')
% saveas(fig, ['results/'  num2str(imageId) '_3d.png']);
% close(fig);
pause(1);
end

