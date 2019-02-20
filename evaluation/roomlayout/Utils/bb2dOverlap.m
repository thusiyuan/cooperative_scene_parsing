function allOverlaps = bb2dOverlap(predictedBbs,groundTruthBbs)

if size(groundTruthBbs,2)==1
    gtbb2D =reshape([groundTruthBbs.gtBb2D],4,[])';
else
    gtbb2D =groundTruthBbs;
end
if size(predictedBbs,2)==1
    predictedBbs =reshape([predictedBbs.gtBb2D],4,[])';
else
    predictedBbs =predictedBbs;
end
gtbb2D(:,3:4) = gtbb2D(:,1:2)+gtbb2D(:,3:4);
predictedBbs(:,3:4) =predictedBbs(:,1:2)+predictedBbs(:,3:4);
allOverlaps =zeros(size(predictedBbs,1),size(gtbb2D,1));

for ngt=1:length(groundTruthBbs)
     allOverlaps(:,ngt) = overlapArray(predictedBbs,gtbb2D(ngt,:));
end
end


function out= overlapArray(bb1,bb2)
    b= [min(bb1(:,3),bb2(3)),min(bb1(:,4),bb2(4))]-...
         [max(bb1(:,1),bb2(1)),max(bb1(:,2),bb2(2))];
    b(b<0)=0;
    intersection =b(:,1).*b(:,2);
    area1=(bb1(:,3)-bb1(:,1)+1).*(bb1(:,4)-bb1(:,2)+1);
    area2=(bb2(:,3)-bb2(:,1)+1).*(bb2(:,4)-bb2(:,2)+1);
    out = intersection./(area1 + area2 - intersection);
    out(bb1(:,1) > bb2(3),:) =0;
    out(bb1(:,2) > bb2(4),:) =0;
    out(bb1(:,3) < bb2(1),:) =0;
    out(bb1(:,4) < bb2(2),:) =0;
end