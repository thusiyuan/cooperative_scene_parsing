function [imageSeg]= getPlanSeg_complete(XYZworldframeTest,SpaceTest,imgSize,imageNum)
% input XYZworldframeTest is alinged in Z
onPlaneThreshold =0.055;
sizethr =750;
normalAgreeThreshold =0.8;

removeNaN = find(~sum(isnan(XYZworldframeTest),2));
XYZworldframeTest = XYZworldframeTest(removeNaN,:);
[zhist,zc]=hist(XYZworldframeTest(:,3),round(20*(SpaceTest.Rz(2)-SpaceTest.Rz(1))));
[~,zpks] =findpeaks([0,zhist,0],'minpeakdistance',3);
zpks = zpks-1;
% figure,
% bar(bincenter,count);
% hold on;
% plot(bincenter(pks),count(pks),'rx');
% remove
range =[max(min(XYZworldframeTest(:,3)),zc(max(1,zpks-1))'),...
        min(zc(min(size(zc,2),zpks+1)),max(XYZworldframeTest(:,3)))';
        zc(end-2),max(XYZworldframeTest(:,3));
        min(XYZworldframeTest(:,3)),zc(3)];
removepts = inrange(XYZworldframeTest(:,3),range);
XYZworldframeTestRemove=XYZworldframeTest;
XYZworldframeTestRemove(removepts>0,:) =[];


% hough lines
[H,T,R] = votelines(XYZworldframeTestRemove(:,1),XYZworldframeTestRemove(:,2));
% blur it 
allangleVote = max(H);
allangleVote = [allangleVote,allangleVote];
shift= 0;
clear rightAngle
for i =1:length(shift)
    rightAngle(i,:) = allangleVote(91+shift(i):180+shift(i))+allangleVote(1:90);
end
[~,AnglepickedLin]= max(rightAngle(:));
[s,Anglepicked]=ind2sub(size(rightAngle),AnglepickedLin);
[~,Rpicked1] = max(H(:,Anglepicked));
[~,Rpicked2] = max(H(:,Anglepicked+90+shift(s)));
P = [R(Rpicked1),T(Anglepicked);R(Rpicked2),T(Anglepicked)+90+shift(s)];


rotation = min(P(:,2));
Rot = getRotationMatrix('z',-1*rotation/180*pi);
XYZworldframeTestNew = [[Rot(1:2,1:2)*XYZworldframeTest(:,[1,2])']', XYZworldframeTest(:,3)];
normals = points2normals(XYZworldframeTestNew);


% find  corner 
[xhist,xc]=hist(XYZworldframeTestNew(:,1),round((SpaceTest.Rx(2)-SpaceTest.Rx(1))/0.05));
[~,xpks] =findpeaks([0,xhist,0],'MINPEAKHEIGHT',50);
xpks = xpks-1;
[yhist,yc]=hist(XYZworldframeTestNew(:,2),round((SpaceTest.Rx(2)-SpaceTest.Rx(1))/0.05));
[~,ypks] =findpeaks([0,yhist,0],'MINPEAKHEIGHT',50);
ypks = ypks-1;
[zhist,zc]=hist(XYZworldframeTestNew(:,3),round((SpaceTest.Rx(2)-SpaceTest.Rx(1))/0.05));
[~,zpks] =findpeaks([0,zhist,0],'MINPEAKHEIGHT',50);
zpks = zpks-1;


%% project points onto this plan and caculate conected component 
gid =1;
Segment ={};
SegmentSize = [];
imageSegFill = zeros(imgSize);
for p =1:size(xpks,2)
    inLine =find(abs(XYZworldframeTestNew(:,1)-xc(xpks(p)))<onPlaneThreshold&abs(normals(1,:)')>normalAgreeThreshold);
    conthisSegMask = zeros(imgSize);
    conthisSegMask(removeNaN(inLine)) =1;
    label = bwlabel(conthisSegMask,8);
    unique_label = unique(label);
    unique_label(unique_label==0)=[];
    for i =1:length(unique_label),
         if sum(label(:)==unique_label(i))>sizethr,
            Segment(gid) = {label==unique_label(i)};
            SegmentSize(gid) = sum(label(:)==unique_label(i));
            imageSegFill(label==unique_label(i))=gid;
            gid = gid+1;
         end
    end
end


for p =1:size(ypks,2)
    inLine =find(abs(XYZworldframeTestNew(:,2)-yc(ypks(p)))<onPlaneThreshold&abs(normals(2,:)')>normalAgreeThreshold);
    conthisSegMask = zeros(imgSize);
    conthisSegMask(removeNaN(inLine)) =1;
    label = bwlabel(conthisSegMask,8);
    unique_label = unique(label);
    unique_label(unique_label==0)=[];
    for i =1:length(unique_label),
         if sum(label(:)==unique_label(i))>sizethr,

            Segment(gid) = {label==unique_label(i)};
            SegmentSize(gid) = sum(label(:)==unique_label(i));
            imageSegFill(label==unique_label(i))=gid;
            gid = gid+1;
         end
    end
end

floorSize = 0;
floorMask = zeros(imgSize);
Lowest = nanmin(XYZworldframeTestNew(:,3));
for p =1:size(zpks,2) 
    inLine =find(abs(XYZworldframeTestNew(:,3)-zc(zpks(p)))<onPlaneThreshold&abs(normals(3,:)')>normalAgreeThreshold);
    conthisSegMask = zeros(imgSize);
    conthisSegMask(removeNaN(inLine)) =1;
    label = bwlabel(conthisSegMask,8);
    unique_label = unique(label);
    unique_label(unique_label==0)=[];
    for i =1:length(unique_label),
         if sum(label(:)==unique_label(i))>sizethr,
            Segment(gid) = {label==unique_label(i)};
            SegmentSize(gid) = sum(label(:)==unique_label(i));
            if p<2&&SegmentSize(gid)>floorSize&&zc(zpks(p))-Lowest<0.25,
                floorSize = SegmentSize(gid);
                floorMask = label==unique_label(i);
            end
            imageSegFill(label==unique_label(i))=gid;
            gid = gid+1;
         end
    end
end
%%

[SegmentSize,segid] = sort(SegmentSize,'descend');
Segment = Segment(segid);
cnt =1;
imageSeg = zeros(imgSize);
for i =1:length(Segment)
    idx = Segment{i}&imageSeg==0;
    se = strel('disk',3);
    idx2 = imclose(idx,se);
    idx(imageSegFill==0&(idx2&~idx))=1;
    if sum(idx(:))>sizethr
       imageSeg(idx)=cnt;
       cnt =cnt+1;
    end
end

while true,
    % remaining points 
    remaining_points =imageSeg==0;
    cntPre = cnt;
    label = bwlabel(remaining_points,8);
    unique_label = unique(label);
    unique_label(unique_label==0)=[];
    directions = icosahedron2sphere(0); 
    directions = directions(directions(:,2) <= 0,:);
    for i =1:length(unique_label)
        remaining_points_conn = label==unique_label(i);
        remaining_points_conn_ptsidx = find(remaining_points_conn(removeNaN));
        XYZworldframeTestRemaining_conn = XYZworldframeTest(remaining_points_conn_ptsidx,:);
        normals_conn = normals(:,remaining_points_conn_ptsidx);
        if size(XYZworldframeTestRemaining_conn,1)>sizethr
            [~, idx] = max(abs(directions * normals_conn),[],1);
            majorDirIdx = mode(idx);
            normalAgreeId = find(idx ==majorDirIdx);
            [B,~,inliers] = ransacfitplane(XYZworldframeTestRemaining_conn(normalAgreeId,:)', onPlaneThreshold);        
            if sum(inliers)>sizethr
                thisSeg = zeros(imgSize);
                thisSeg(removeNaN(remaining_points_conn_ptsidx(normalAgreeId(inliers)))) =1;
                thisSeg_label = bwlabel(thisSeg,8);
                thisSeg_unique_label = unique(label);
                thisSeg_unique_label(thisSeg_unique_label==0)=[];
                for j = 1:length(thisSeg_unique_label),
                    if sum(thisSeg_label(:)==thisSeg_unique_label(j))>sizethr,
                        idxSeg = thisSeg_label==thisSeg_unique_label(j);
                        idxSeg = imclose(idxSeg,se);
                        imageSeg(idxSeg) =cnt;
                        cnt=cnt+1;
                    end
                end
                
            end
        end
    end
    if cntPre==cnt,
        break;
    end
end
map= [0,randperm(max(imageSeg(:)+1))];
map(map==1) =[];
imageSeg = map(imageSeg+1);
if sum(floorMask(:))>0,
    floorMask = imclose(floorMask,se);
    imageSeg(floorMask) = 1;
end
figure(1),imagesc(imageSeg);
figure(2),imagesc(floorMask);
%Boudary = [D1 D2 D3 D4];
if imageNum> 0, 
    im = getImagesc(imageSeg);
    mkdir(segpath);
    imwrite(im,sprintf('%s/%04d.jpg',segpath,imageNum));
    
    im = getImagesc(floorMask);
    imwrite(im,sprintf('%s/%04d_floor.jpg',segpath,imageNum));
    
    save(sprintf('%s/%04d.mat',segpath,imageNum),'imageSeg','floorMask')
    
end
end

function [hough_transform,T,R] = votelines(X,Y)
         thetaResolution = 1;
         rhoResolution = 0.1;
         T = [1:180];
         theta = T/180*pi;
         rho = X(:)*cos(theta)+ Y(:)*sin(theta);
         % quantize the rho 
         rhoNorm = max(1,round((rho-min(rho(:)))/rhoResolution));
         R =[1:max(rhoNorm(:))]*rhoResolution+min(rho(:));
         hough_transform = zeros(max(rhoNorm(:)),size(T,2));
         TT = repmat(T,[size(rhoNorm,1),1]);
         hough_transform = accumarray([rhoNorm(:),TT(:)],1,size(hough_transform));
end
