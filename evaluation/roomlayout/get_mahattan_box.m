clear all
clc
close all
opts = getOpts();
addpath('mhUtils'); % only needed for manhattan box
addpath('Utils');
load('SUNRGBDMetaUS.mat')
for i = 9001:10335
    disp(i)
    data = SUNRGBDMeta(i);
    % predicted room
    disp('computing ManhattanBox room estimate');
    manhattan_layout = ManhattanBox(data,opts);
    mat_path = ['/home/siyuan/Documents/nips2018/sunrgbd/3dlayout/' int2str(i) '.mat'];
    save(mat_path, 'manhattan_layout')
end

