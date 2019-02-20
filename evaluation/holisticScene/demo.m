clc;
clear all;

toolboxpath = '/home/siyuan/Documents/Dataset/SUNRGBD_ALL/SUNRGBDtoolbox/SUNRGBDtoolbox';
addpath(genpath(toolboxpath));
load(fullfile(toolboxpath,'/traintestSUNRGBD/allsplit.mat'));
load(fullfile('SUNRGBDMetaUS.mat'));
new_3d = load('SUNRGBDMeta3DBB_v2.mat');
cls = {'bathtub','bed','bookshelf','box','chair','counter','desk','door','dresser','garbage_bin','lamp','monitor','night_stand','pillow','sink','sofa','table','tv','toilet'};
thresh_iou = [0.01,0.05:0.05:1.6,0.8:0.2:1];
result_path = '/home/siyuan/Documents/cvpr2018/evaluation';
vis = true;

%% evaluation
iou_precision =[];
iou_recall =[];
label_precision =[];
iouHolistic_all = []; 
for imageId = 11
    % fake result always use same room and object boxes
    data = SUNRGBDMeta(imageId);
    bdb_3d = load(fullfile(result_path, int2str(imageId), 'bdb_3d.mat'));
    bdb = bdb_3d.bdb;
    layout = load(fullfile(result_path, int2str(imageId), 'layout.mat'));
    predictRoom = layout.layout;
    gt_bdb = new_3d.SUNRGBDMeta(imageId).groundtruth3DBB;
    % evaluate
    for i = 1:size(bdb, 1)
    	predictedBbs(i) = create_bounding_box_3d([bdb(i, 9), bdb(i, 10); bdb(i, 11), bdb(i, 12)], [bdb(i, 3), bdb(i, 4), bdb(i, 5)], [bdb(i, 6), bdb(i, 7), bdb(i, 8)]);
    end
    eval_result = eval_holisticScencePR(predictedBbs',gt_bdb',thresh_iou,cls); 
    iouHolistic = eval_holisticScenceIoU(data,data.gtCorner3D,gt_bdb',predictRoom,predictedBbs',vis);
    
    iou_precision = [iou_precision;eval_result.iou_precision];
    iou_recall = [iou_recall;eval_result.iou_recall];
    label_precision = [label_precision;eval_result.label_precision];
    iouHolistic_all = [iouHolistic_all iouHolistic];
end
%% plot and average
figure;
thresh_iou = [0.01,0.05:0.05:1.6,0.8:0.2:1];
load('iou_precision.mat');
load('iou_recall.mat');
load('iouHolistic_all.mat');
plot(thresh_iou,mean(iou_precision),'r','LineWidth',3);
hold on;
plot(thresh_iou,mean(iou_recall),'g','LineWidth',3);
plot(thresh_iou,mean(label_precision),'b','LineWidth',3);
legend({'geometric precision','geometric recall','recognition recall'},'FontSize',27)
%xlabel('IoU threshold');
axis([0 1 0 1])
iou_precision_average =  mean(iou_precision(:,6));
iou_recall_average =  mean(iou_recall(:,6));
label_precision_average =  100*mean(label_precision(:,6));
iouHolistic_average = mean(iouHolistic_all);
sprintf('Average:\n geometric precision: %.3f\n geometric recall: %.3f\n recognition recall: %.3f\n %.3f\n',...
    iou_precision_average,iou_recall_average,label_precision_average,iouHolistic_average)