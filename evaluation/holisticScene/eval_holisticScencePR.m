function eval_result=eval_holisticScencePR(predictedBbs,groundTruthBbs,thresh_iou,cls)
    % input : 
    % cls : object categorys that in consideration 
    % thresh_iou: iou threshold for detection with ground truth
    % ouput : 
    % label_precision: measure the recall of recognition for both semantics and geometry
    % iou_recall : measure the geometric prediction recall
    % iou_precision : measure the geometric prediction precision
    if ~exist('thresh_iou','var')
        thresh_iou = [0.01,0.05:0.05:0.6,0.8:0.2:1];
    end
    if ~exist('cls','var')
        cls = {'bathtub','bed','bookshelf','box','chair','counter','desk','door','dresser','garbage_bin','lamp','monitor','night_stand','pillow','sink','sofa','table','tv','toilet'};
    end
    
    allOverlaps = bb3dOverlapCloseForm(predictedBbs,groundTruthBbs);
    matchpg =[];
    maxOverlaps =[];
    while sum(allOverlaps(:)>0)>0
        [os,ind] = max(allOverlaps(:));
        [i,j] = ind2sub(size(allOverlaps),ind);
        matchpg(end+1,:) =[i,j];
        maxOverlaps(end+1) = os;
        allOverlaps(i,:) =0;
        allOverlaps(:,j) =0;
    end
    
   
    iou_recall = zeros(1,length(thresh_iou));
    iou_precision = zeros(1,length(thresh_iou));
    label_precision = zeros(1,length(thresh_iou));
    label_recall = zeros(1,length(thresh_iou));
    if ~isempty(matchpg)
        for i = 1:length(thresh_iou)
            iou_match = find(maxOverlaps>thresh_iou(i));
            pclass = [predictedBbs(matchpg(iou_match,1)).classid];
            [~,gtclass] = ismember({groundTruthBbs(matchpg(iou_match,2)).classname},cls);
            if length(pclass) > 0
                pclass = double(pclass);
            end
            correctLabel = sum([pclass-gtclass]==0);
            label_precision(i) = correctLabel/length(predictedBbs);
            label_recall(i) = correctLabel/length(groundTruthBbs);
            iou_recall(i) = length(iou_match)/length(groundTruthBbs);
            iou_precision(i) = length(iou_match)/length(predictedBbs);
        end
    end
    eval_result = struct('label_precision',label_precision,'label_recall',label_recall,'iou_recall',iou_recall,'iou_precision',iou_precision);
end