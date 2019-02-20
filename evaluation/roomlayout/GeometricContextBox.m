function outCorner3D = GeometricContextBox(sequenceName,visualization)


if ~exist('sequenceName','var')
    %sequenceName = 'NYUv2images/NYU1385';
    %groundTruthName  ='http://sun3d.cs.princeton.edu/data/NYUv2images/NYU1385/annotation3Dlayout/index.json';


    %sequenceName = '/n/fs/sun3d/data/b3do/img_0691';
    %sequenceName ='/n/fs/sun3d/data/kinect2data/002803_2014-06-22_19-53-26_094959634447_rgbf000095-resize/';

    %sequenceName ='/n/fs/sun3d/data/kinect2data/001121_2014-06-15_18-20-28_260595134347_rgbf000134-resize/';
    %groundTruthName  ='http://sun3d.cs.princeton.edu/data/rgbd_voc/001121_2014-06-15_18-20-28_260595134347_rgbf000134-resize/annotation3Dlayout/index.json';

    sequenceName = 'kinect2data/001121_2014-06-15_18-20-28_260595134347_rgbf000134-resize';
    %groundTruthName  ='http://sun3d.cs.princeton.edu/data/rgbd_voc/001121_2014-06-15_18-20-28_260595134347_rgbf000134-resize/annotation3Dlayout/index.json';
end

%groundTruthName  = [ 'http://sun3d.cs.princeton.edu/data/' sequenceName '/annotation3Dlayout/index.json'];

if ~exist('visualization','var')
    visualization = false;
end

%{
addpath /n/fs/modelnet/slidingShape_release_all/code_benchmark
initPath;
addpath /n/fs/modelnet/SUN3DV2/prepareGT
setup_benchmark;
%}

%% Annotator rotation matrix

extrinsicsFiles = dirSmart(fullfile('http://sun3d.cs.princeton.edu/data/',sequenceName,'extrinsics/'),'txt');
extrinsicsC2W = permute(reshape(readValuesFromTxt(fullfile('http://sun3d.cs.princeton.edu/data/',sequenceName,'extrinsics',extrinsicsFiles(end).name)),4,3,[]),[2 1 3]);


data=readframe(fullfile('/n/fs/sun3d/data/',sequenceName));
im = imread(data.rgbpath);


GCresult = '/n/fs/lsun/SpatialLayout/spatiallayoutcode/Result';
fnameMap = '0001.mat';



fnameMapping = load('/Volumes/modelnet/slidingShape_release_all/code_benchmark/list.mat');

seqMapping = {};

for i=1:length(fnameMapping.imagepath)
    nn = fnameMapping.imagepath{i}(18:end);
    endid = findstr(nn,'/image/');
    nn = nn(1:endid-1);
    if nn(end)=='/'
        nn = nn(1:end-1);
    end
    seqMapping{end+1} = nn;
end

id = find(ismember(seqMapping,sequenceName));


fname = fullfile(GCresult, sprintf('%.4d.mat',id));
load(fname);

polyg = boxlayout.polyg(boxlayout.reestimated(1,2),:);



if visualization
    figure
    imshow(im);
    for i=1:5
        if ~isempty(polyg{i})
            hold on;
            plot(polyg{i}(:,1)*2,polyg{i}(:,2)*2,'-r','LineWidth',3);
            plot(polyg{i}(:,1)*2,polyg{i}(:,2)*2,'*b');
        end
    end
    axis tight
end

floor = polyg{1}'*2;

h= size(im,1);
w= size(im,2);

%point2reconstruct = floor(2,:) < h-10;
%indices = find(point2reconstruct==0);

floor3D = zeros(3,size(floor,2));

for i=1:size(floor,2)
    Xcamera =[floor(1,i)-data.K(1,3); floor(2,i)-data.K(2,3); data.K(1,1)];    
    Xworld = extrinsicsC2W(1:3,1:3) * Xcamera;
    floor3D(:,i) = Xworld / Xworld(2);
end


%% determine the scale


[rgb,points3d,imgZ]=read3dPoints(data);
[x,y] = meshgrid(1:size(imgZ,2), 1:size(imgZ,1));
XYZcamera = zeros(size(imgZ,1),size(imgZ,2),3);
XYZcamera(:,:,1) = (x-data.K(1,3)).*imgZ/data.K(1,1);
XYZcamera(:,:,2) = (y-data.K(2,3)).*imgZ/data.K(2,2);
XYZcamera(:,:,3) = imgZ;
XYZcamera = reshape(XYZcamera,[],3)';
XYZcamera = XYZcamera(:,XYZcamera(3,:)~=0);
XYZworld = extrinsicsC2W(1:3,1:3) * XYZcamera;


%{
bestScale = [];
for scale=1:0.1:2
    
    in = XYZworld(2,:)<scale;
    
    inPoly = inpolygon(XYZworld(1,:),XYZworld(3,:),floor3D(1,:)*scale,floor3D(3,:)*scale);
    
    in = in & inPoly;
    
    ratio = sum(in)/size(XYZworld,2);
    fprintf('scale = %f   ratio = %f\n', scale, ratio)
    if ratio>0.75
        bestScale = scale;
        break;
    end
    
    if visualization
        figure
        plot3(0,0,0,'*b');
        hold on;
        plot3(floor3D(1,[1:end 1])*scale,floor3D(2,[1:end 1])*scale,floor3D(3,[1:end 1])*scale,'-r','LineWidth',3);
        axis equal;
        xlabel('x');
        ylabel('y');
        zlabel('z');    
        hold on
        plot3(XYZworld(1,:),XYZworld(2,:),XYZworld(3,:),'.k');   
        hold on;
        %XYZworld = XYZworld(:,1:10:end);
        plot3(XYZworld(1,in),XYZworld(2,in),XYZworld(3,in),'.b');    
        axis equal;
    end
end

if isempty(bestScale)
    bestScale = 1.6;
end
%}

bestScale = prctile(XYZworld(2,:),99.9);


floor3D = floor3D * bestScale;

floor3D(:,end+1) = [0;bestScale;0];

ind = convhull(floor3D(1,:),floor3D(3,:));

floor3D = floor3D(:,ind);

yMin = -4;

outCornerMax = floor3D;
outCornerMax(2,:) = yMin;
outCorner3D = [floor3D outCornerMax];


if visualization

    figure
    plot3(0,0,0,'*b');
    hold on;
    plot3(floor3D(1,[1:end 1]),floor3D(2,[1:end 1]),floor3D(3,[1:end 1]),'-r','LineWidth',3);
    axis equal;
    xlabel('x');
    ylabel('y');
    zlabel('z');    

    hold on;
    %XYZworld = XYZworld(:,1:10:end);
    plot3(XYZworld(1,:),XYZworld(2,:),XYZworld(3,:),'.k');    
    axis equal;
end


if visualization
    figure
    numCorners = size(outCorner3D,2)/2;
    patch(outCorner3D(1,1:numCorners),outCorner3D(2,1:numCorners),outCorner3D(3,1:numCorners),'k');
    hold on
    patch(outCorner3D(1,numCorners+1:end),outCorner3D(2,numCorners+1:end),outCorner3D(3,numCorners+1:end),'b');
    hold on
    for i=1:numCorners-1
        patch(outCorner3D(1,[i i+1 numCorners+i+1 numCorners+i]),outCorner3D(2,[i i+1 numCorners+i+1 numCorners+i]),outCorner3D(3,[i i+1 numCorners+i+1 numCorners+i]),'r');
        hold on
    end
    patch(outCorner3D(1,[numCorners 1  numCorners+1 numCorners*2]),outCorner3D(2,[numCorners 1  numCorners+1 numCorners*2]),outCorner3D(3,[numCorners 1  numCorners+1 numCorners*2]),'r');
    axis equal
    alpha(0.6);
    title('Prediction Result');
    [x,y] = meshgrid(1:size(imgZ,2), 1:size(imgZ,1));
    XYZcamera = zeros(size(imgZ,1),size(imgZ,2),3);
    XYZcamera(:,:,1) = (x-data.K(1,3)).*imgZ/data.K(1,1);
    XYZcamera(:,:,2) = (y-data.K(2,3)).*imgZ/data.K(2,2);
    XYZcamera(:,:,3) = imgZ;
    XYZcamera = reshape(XYZcamera,[],3)';
    XYZcamera = XYZcamera(:,XYZcamera(3,:)~=0);
    XYZworld = extrinsicsC2W(1:3,1:3) * XYZcamera;
    XYZworld = XYZworld(:,1:10:end);
    hold on;
    plot3(XYZworld(1,:),XYZworld(2,:),XYZworld(3,:),'.k');
    
end
