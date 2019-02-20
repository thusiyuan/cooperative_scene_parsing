function [minZ,maxZ,minX,maxX,minY,maxY] = getRoomBox(data,show)

addpath /n/fs/modelnet/slidingShape_release_all/code_benchmark
initPath;
addpath /n/fs/modelnet/SUN3DV2/prepareGT
setup_benchmark;

%sequenceName = 'NYUv2images/NYU1385';
%groundTruthName  ='http://sun3d.cs.princeton.edu/data/NYUv2images/NYU1385/annotation3Dlayout/index.json';
%sequenceName = '/n/fs/sun3d/data/b3do/img_0691';
%sequenceName ='/n/fs/sun3d/data/kinect2data/002803_2014-06-22_19-53-26_094959634447_rgbf000095-resize/';
%sequenceName ='/n/fs/sun3d/data/kinect2data/001121_2014-06-15_18-20-28_260595134347_rgbf000134-resize/';
%groundTruthName  ='http://sun3d.cs.princeton.edu/data/rgbd_voc/001121_2014-06-15_18-20-28_260595134347_rgbf000134-resize/annotation3Dlayout/index.json';
%sequenceName = 'kinect2data/001121_2014-06-15_18-20-28_260595134347_rgbf000134-resize';
%data=readframe(fullfile('/n/fs/sun3d/data/',sequenceName));


%%

[rgb,points3d,imgZ]=read3dPoints(data);
Space=struct('Rx',[nanmin(points3d(:,1)), nanmax(points3d(:,1))],...
             'Ry',[nanmin(points3d(:,2)),nanmax(points3d(:,2))],...
             'Rz',[nanmin(points3d(:,3)),nanmax(points3d(:,3))],'s',0.1);

imgSize= size(imread(data.depthpath));
[imageSeg,Rot]= getPlanSeg(points3d,Space,imgSize);

points3d_align = [[Rot(1:2,1:2)*points3d(:,[1,2])']', points3d(:,3)];

mhaX = reshape(points3d_align(:,1),size(imgZ));
mhaY = reshape(points3d_align(:,2),size(imgZ));
mhaZ = reshape(points3d_align(:,3),size(imgZ));

numSeg = max(imageSeg(:));
imageSeg(imageSeg(:)==0)= numSeg+1;
thresholdArea = 0.001;


for segID=1:numSeg
    roi = imageSeg==segID;
    
    if sum(roi(:)) > thresholdArea * numel(imageSeg)
        planeX = mhaX(roi(:));  
        planeY = mhaY(roi(:));  
        planeZ = mhaZ(roi(:));  
        stdXYZ =[std(planeX,1) std(planeY,1) std(planeZ,1)];
        [~,minIdx(segID)]=min(stdXYZ);
    else
        minIdx(segID) = 0;
    end
end
minIdx(numSeg+1)=0;

%figure, imagesc(minIdx(imageSeg));
%cmap=[0.5 0.5 0.5; 1 0 0; 0 1 0; 0 0 1];
%colormap(cmap)

roi = minIdx(imageSeg)==1;
maxX = max(mhaX(roi(:)));
minX = min(-1,min(mhaX(roi(:))));

roi = minIdx(imageSeg)==2;
maxY = max(mhaY(roi(:)));
minY = min(-1,min(mhaY(roi(:))));

roi = minIdx(imageSeg)==3;
maxZ = max(mhaZ(roi(:)));
minZ = min(0,min(mhaZ(roi(:))));

%% visualization 
if show
floor = [...
minX minY minZ
maxX minY minZ
maxX maxY minZ
minX maxY minZ]';

wallX = [...
maxX minY minZ
maxX maxY minZ
maxX maxY maxZ
maxX minY maxZ]';

wallY = [...
minX maxY minZ
maxX maxY minZ
maxX maxY maxZ
minX maxY maxZ]';


figure
clf
vis_point_cloud(points3d_align,rgb,10,10000); 
xlabel('x'); ylabel('y'); zlabel('z');
hold on; patch(wallX(1,:),wallX(2,:),wallX(3,:),'r')
hold on; patch(wallY(1,:),wallY(2,:),wallY(3,:),'g')
hold on; patch(floor(1,:),floor(2,:),floor(3,:),'b')
alpha(0.6);
end
%}