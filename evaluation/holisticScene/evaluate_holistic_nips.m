clc;
clear all;

toolboxpath = '/home/siyuan/Documents/Dataset/SUNRGBD_ALL/SUNRGBDtoolbox/SUNRGBDtoolbox';
addpath(genpath(toolboxpath));
load(fullfile(toolboxpath,'/traintestSUNRGBD/allsplit.mat'));
old_3d = load('SUNRGBDMetaUS.mat');
new_3d = load('SUNRGBDMeta3DBB_v2.mat');
% cls = {'bathtub','bed','bookshelf','box','chair','counter','desk','door','dresser','garbage_bin','lamp','monitor','night_stand','pillow','sink','sofa','table','tv','toilet'};
cls = {'recycle_bin', 'cpu', 'paper', 'toilet', 'stool', 'whiteboard', 'coffee_table', 'picture', 'keyboard', 'dresser', 'painting', 'bookshelf', 'night_stand', 'endtable', 'drawer', 'sink', 'monitor', 'computer', 'cabinet', 'shelf', 'lamp', 'garbage_bin', 'box', 'bed', 'sofa', 'sofa_chair', 'pillow', 'desk', 'table', 'chair'};
thresh_iou = [0.01,0.05:0.05:0.6,0.8:0.2:1];
result_path = '/home/siyuan/Documents/nips2018/sunrgbd/results_full';
vis = false;
listing = dir(result_path);
set = listing(3:end);
len = length(set);
parpool(4);
parfor imageId = 1:5050
    data = old_3d.SUNRGBDMeta(imageId);
    if ~exist(fullfile(result_path, int2str(imageId), 'bdb_3d.mat'), 'file');
        continue;
    end
    if size(data.gtCorner3D, 2) == 0
        continue;
    end
    disp(imageId);
    bdb_3d = load(fullfile(result_path, int2str(imageId), 'bdb_3d.mat'));
    bdb = bdb_3d.bdb;
    layout = load(fullfile(result_path, int2str(imageId), 'layout.mat'));
    predictRoom = layout.layout;
    gt_bdb = new_3d.SUNRGBDMeta(imageId).groundtruth3DBB;
    % evaluate
    predictedBbs = [];
    for i = 1:length(bdb)
        predictedBbs= [predictedBbs bdb{i}];
    end
    eval_result = eval_holisticScencePR(predictedBbs',new_3d.SUNRGBDMeta(imageId).groundtruth3DBB',thresh_iou,cls); 
    iouHolistic = eval_holisticScenceIoU(data,data.gtCorner3D,new_3d.SUNRGBDMeta(imageId).groundtruth3DBB',predictRoom',predictedBbs',vis);
    
    iou_precision{imageId} = eval_result.iou_precision;
    iou_recall{imageId} = eval_result.iou_recall;
    label_recall{imageId} = eval_result.label_recall;
    label_precision{imageId} = eval_result.label_precision;
    iouHolistic_all{imageId} = iouHolistic;
    disp(iouHolistic);
end
%% plot and average
iou_precision = cell2mat(iou_precision(:));
iou_recall = cell2mat(iou_recall(:));
label_precision = cell2mat(label_precision(:));
label_recall = cell2mat(label_recall(:));
iouHolistic_all = cell2mat(iouHolistic_all(:));
%% plot and average
figure,
plot(thresh_iou,mean(iou_precision()),'r','LineWidth',3);
hold on;
plot(thresh_iou,mean(iou_recall),'g','LineWidth',3);
plot(thresh_iou,mean(label_recall),'b','LineWidth',3);
plot(thresh_iou, mean(label_precision), 'k', 'LineWidth', 3);
legend({'geometric precision','geometric recall','recognition recall', 'recognition precision'},'FontSize',27)
%xlabel('IoU threshold');
axis([0 1 0 1])
iou_precision_average =  mean(iou_precision(:,4));
iou_recall_average =  mean(iou_recall(:,4));
label_recall_average =  100*mean(label_recall(:,4));
label_precision_average = 100 * mean(label_precision(:, 4));
iouHolistic_all = iouHolistic_all(~isnan(iouHolistic_all));
iouHolistic_average = mean(iouHolistic_all);
sprintf('Average:\n geometric precision: %.3f\n geometric recall: %.3f\n recognition recall: %.3f\n %.3f\n',...
    iou_precision_average,iou_recall_average,label_recall_average,iouHolistic_average)

