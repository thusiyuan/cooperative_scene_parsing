function [imageSeg,Rot]= getPlanSeg(XYZworldframeTest,SpaceTest,imgSize)
% input XYZworldframeTest is alinged in Z
%load([ '/n/fs/modelnet/NYUdataSet/NYUdatafeatureNew/' 'feature_' num2str(imageNum) '.mat'],'XYZworldframeTest','rgbTest','SpaceTest','imgDepth');    
onPlaneThreshold =0.055;
sizethr =500;
normalAgreeThreshold =0.8;
removeNaN = find(~sum(isnan(XYZworldframeTest),2));
XYZworldframeTest = XYZworldframeTest(removeNaN,:);
[zhist,zc]=hist(XYZworldframeTest(:,3),round(20*(SpaceTest.Rz(2)-SpaceTest.Rz(1))));
[~,zpks] =findpeaks([0,zhist,0],'minpeakdistance',3);
zpks = zpks-1;

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
[yhist,yc]=hist(XYZworldframeTestNew(:,2),round((SpaceTest.Ry(2)-SpaceTest.Ry(1))/0.05));
[~,ypks] =findpeaks([0,yhist,0],'MINPEAKHEIGHT',50);
ypks = ypks-1;
[zhist,zc]=hist(XYZworldframeTestNew(:,3),round((SpaceTest.Rz(2)-SpaceTest.Rz(1))/0.05));
[~,zpks] =findpeaks([0,zhist,0],'MINPEAKHEIGHT',50);
zpks = zpks-1;

%project points onto this plan and caculate conected component 

gid =1;
imageSeg = zeros(imgSize);
for p =1:size(xpks,2)
    inLine =find(abs(XYZworldframeTestNew(:,1)-xc(xpks(p)))<onPlaneThreshold&abs(normals(1,:)')>normalAgreeThreshold);
    conthisSegMask = zeros(imgSize);
    conthisSegMask(removeNaN(inLine)) =1;
    label = bwlabel(conthisSegMask,8);
    unique_label = unique(label);
    unique_label(unique_label==0)=[];
    for i =1:length(unique_label),
         if sum(label(:)==unique_label(i))>sizethr,
            imageSeg(label==unique_label(i))=gid;
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
            imageSeg(label==unique_label(i))=gid;
            gid = gid+1;
         end
    end
end

for p =1:size(zpks,2) 
    inLine =find(abs(XYZworldframeTestNew(:,3)-zc(zpks(p)))<onPlaneThreshold&abs(normals(3,:)')>normalAgreeThreshold);
    conthisSegMask = zeros(imgSize);
    conthisSegMask(removeNaN(inLine)) =1;
    label = bwlabel(conthisSegMask,8);
    unique_label = unique(label);
    unique_label(unique_label==0)=[];
    for i =1:length(unique_label),
         if sum(label(:)==unique_label(i))>sizethr,
            imageSeg(label==unique_label(i))=gid;
            gid = gid+1;
         end
    end
end
%%

% rotate back 
%{
corner_r = [xc(xpks(1)) yc(ypks(1));xc(xpks(end)) yc(ypks(end))];
corner_r = get4corner(corner_r);
corner = [[Rot(1:2,1:2)'*corner_r(:,[1:2])']'];
D1 = cos(P(1,2)*pi/180)*corner(1,1)+sin(P(1,2)*pi/180)*corner(1,2);
D2 = cos(P(2,2)*pi/180)*corner(1,1)+sin(P(2,2)*pi/180)*corner(1,2);
D3 = cos(P(1,2)*pi/180)*corner(3,1)+sin(P(1,2)*pi/180)*corner(3,2);
D4 = cos(P(2,2)*pi/180)*corner(3,1)+sin(P(2,2)*pi/180)*corner(3,2);
P_new = [D1,P(1,2);D2,P(2,2);D3,P(1,2);D4,P(2,2)];
f = figure, 
vis_point_cloud(XYZworldframeTest,rgbTest,10,5000);hold on;
for j =1:3 
    plot3(corner([j,j+1],1),corner([j,j+1],2),[max(XYZworldframeTest(:,3));max(XYZworldframeTest(:,3))],'-xr','LineWidth',10)
end
plot3(corner([1,4],1),corner([1,4],2),[max(XYZworldframeTest(:,3));max(XYZworldframeTest(:,3))],'-xr','LineWidth',10)

for i =1:gid
    hold on;
    plot3(XYZworldframeTestOrg(imageSeg(:)==i,1),XYZworldframeTestOrg(imageSeg(:)==i,2),XYZworldframeTestOrg(imageSeg(:)==i,3),'+','Color',rand([1,3]));
end

axis equal;
axis tight;
view(30,50)
saveas(f,['./result/' num2str(imageNum) '.fig']);
saveas(f,['./result/' num2str(imageNum) '.jpg']);

for gid =1:length(gropuind)
    imageSeg(removeNaN(gropuind{gid})) = gid;
end
%}



%figure(1),imagesc(imageSeg)
%Boudary = [D1 D2 D3 D4];
%{
if imageNum> 0, 
    im = getImagesc(imageSeg);
    mkdir(segpath);
    imwrite(im,sprintf('%s/%04d.jpg',segpath,imageNum));
end
%}
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
