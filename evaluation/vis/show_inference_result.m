%% show our result
clc;
close all;
clear all;
addpath(genpath('.'))
result_path = '/home/siyuan/Documents/nips2018/sunrgbd/results_test';
%% change path
figure(1);
for imageId = 12 %480
bdb_3d = {};
bdb_2d = {};
disp(imageId);
bdb_result_path = fullfile(result_path, int2str(imageId), 'bdb_3d.mat');
layout_path = fullfile(result_path, int2str(imageId), 'layout.mat');
if ~exist(bdb_result_path, 'file')
    continue;
end
r_path = fullfile(result_path, int2str(imageId), 'r_ex.mat');
load(r_path);
load(bdb_result_path);
r_ex
%%
for i = 1:size(bdb, 2)
    bdb_3d{i} = bdb{1, i};
end
roomLayout = load(layout_path);

K = [529.5,0,365;0,529.5,265;0,0,1];

% get 2d corners
for kk = 1:length(bdb_3d)
    bdb_2d_temp = get_corners_of_bb3d(bdb_3d{kk});
    bdb_2d_temp = bdb_2d_temp(:, [1, 3, 2]);
    bdb_2d_temp(:, 2) = - bdb_2d_temp(:, 2);
    bdb_2d_temp = (K*r_ex*bdb_2d_temp')';
    bdb_2d_corner(:, 1) = bdb_2d_temp(:, 1) ./ bdb_2d_temp(:, 3);
    bdb_2d_corner(:, 2) = bdb_2d_temp(:, 2) ./ bdb_2d_temp(:, 3);
    bdb_2d{kk} = bdb_2d_corner;
end

%% draw 2D
imshow(xxx);
hold on; 
for kk =1:length(bdb_2d)
% for kk = [1,2,4,5]
    draw_square_3d(bdb_2d{kk}, myObjectColor(kk), 5);
end
hold off;
set(gca,'xtick',[])
set(gca,'ytick',[])
axis off;
%saveas(gcf, ['imgs_full/' num2str(imageId) '_result.png']);
% clf;


%% draw 3D 
figure(2);
% for kk =1:length(data.groundtruth3DBB)
%    vis_cube(data.groundtruth3DBB(kk),'r')
% end
for kk = 1:length(bdb_3d)
    vis_cube(bdb_3d{kk}, myObjectColor(kk), 2);
end
Linewidth = 3;
maxhight = 1.2;
drawRoom(roomLayout.layout','b',Linewidth,maxhight);
hold off;
set(gca,'xtick',[])
set(gca,'ytick',[])
set(gca, 'ztick', [])
axis off;
end
%%

% hold on
% outCorner3D = roomLayout.layout';
% numCorners = size(outCorner3D,2)/2;
% patch(outCorner3D(1,1:numCorners),outCorner3D(2,1:numCorners),outCorner3D(3,1:numCorners),'k');
% hold on
% patch(outCorner3D(1,numCorners+1:end),outCorner3D(2,numCorners+1:end),outCorner3D(3,numCorners+1:end),'b');
% hold on
% for i=1:numCorners-1
%     patch(outCorner3D(1,[i i+1 numCorners+i+1 numCorners+i]),outCorner3D(2,[i i+1 numCorners+i+1 numCorners+i]),outCorner3D(3,[i i+1 numCorners+i+1 numCorners+i]),'r');
%     hold on
% end
% patch(outCorner3D(1,[numCorners 1  numCorners+1 numCorners*2]),outCorner3D(2,[numCorners 1  numCorners+1 numCorners*2]),outCorner3D(3,[numCorners 1  numCorners+1 numCorners*2]),'r');
% axis equal
% alpha(0.2);
% title('Prediction Result');

    %figure
% numCorners = size(gtCorner3D,2)/2;
% patch(gtCorner3D(1,1:numCorners),gtCorner3D(2,1:numCorners),gtCorner3D(3,1:numCorners),'g');
% hold on
% patch(gtCorner3D(1,numCorners+1:end),gtCorner3D(2,numCorners+1:end),gtCorner3D(3,numCorners+1:end),'g');
% hold on
% for i=1:numCorners-1
%     patch(gtCorner3D(1,[i i+1 numCorners+i+1 numCorners+i]),gtCorner3D(2,[i i+1 numCorners+i+1 numCorners+i]),gtCorner3D(3,[i i+1 numCorners+i+1 numCorners+i]),'g');
%     hold on
% end
% patch(gtCorner3D(1,[numCorners 1  numCorners+1 numCorners*2]),gtCorner3D(2,[numCorners 1  numCorners+1 numCorners*2]),gtCorner3D(3,[numCorners 1  numCorners+1 numCorners*2]),'g');
% axis equal
% alpha(0.6);    
% 
%     
% hold on;
% plot3(gridXYZ(1,:),gridXYZ(2,:),gridXYZ(3,:),'.'); xlabel('x'); ylabel('y'); zlabel('z');

