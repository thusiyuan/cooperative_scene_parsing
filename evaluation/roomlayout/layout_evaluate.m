clc;
clear all;
addpath('Utils');
parpool(4);
result_path = '../../metadata/sunrgbd/results_full';
%%
SUNRGBD = load('SUNRGBDMetaUS.mat');
opts.visualization = 0;
listing = dir(result_path);
set = listing(3:end);
len = length(set);
iou = zeros(len, 1);
parfor i = 1:len %% 
    sequence_id = set(i).name;
    layout_path = fullfile(result_path, sequence_id, 'layout.mat');
    if ~exist(layout_path, 'file')
        iou(i) = nan;
        continue
    end
    layout = load(layout_path);
%     if layout.if_l_b == 0
%         iou(i) = nan;
%         continue
%     end
    manhattan_layout = load(['../../metadata/sunrgbd/3dlayout/' sequence_id '.mat']);
    data = SUNRGBD.SUNRGBDMeta(str2num(sequence_id));
    if size(data.gtCorner3D, 2) == 0
        iou(i) = nan;
        continue
    end
    
    iou_temp = evaluateRoom3D(data,data.gtCorner3D,layout.layout',opts);
    if iou(i) == 0
        disp(sequence_id)
    end
    iou(i) = iou_temp; 
    disp(iou(i));
    
end

%%
iou = iou(iou~=0); %% some of the annotation of the 3D layout is wrong, which cause a zero value of valid groundtruth
iou = iou(~isnan(iou));
disp(mean(iou));
save('iou_init.mat', 'iou');