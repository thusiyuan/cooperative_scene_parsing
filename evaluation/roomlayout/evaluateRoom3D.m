function iou = evaluateRoom3D(data,gtCorner3D,outCorner3D,opts,cutoff,delta)

%% set parameters

% set cutoff distance
if ~exist('cutoff','var')
    cutoff = 5.5;
end

% granuality
if ~exist('delta','var')
    delta  = 0.1;
end

%if ~exist('visualization','var')
%    visualization = false;
%end


%% evaluate the accuracy
rangeX = min(min(outCorner3D(1,:)),min(gtCorner3D(1,:)))-delta:delta:max(max(outCorner3D(1,:)),max(gtCorner3D(1,:)))+delta;
rangeY = min(min(outCorner3D(2,:)),min(gtCorner3D(2,:)))-delta:delta:max(max(outCorner3D(2,:)),max(gtCorner3D(2,:)))+delta;
rangeZ = min(min(outCorner3D(3,:)),min(gtCorner3D(3,:)))-delta:delta:max(max(outCorner3D(3,:)),max(gtCorner3D(3,:)))+delta;
[gridX, gridY, gridZ] = ndgrid(rangeX,rangeY,rangeZ);
gridXYZ = [gridX(:)'; gridY(:)'; gridZ(:)'];
clear gridX gridY gridZ
%plot3(gridXYZ(1,:),gridXYZ(2,:),gridXYZ(3,:),'.');


imgSize=size(imread(data.depthpath));

%extrinsicsFiles = dirSmart(fullfile('http://sun3d.cs.princeton.edu/data/',sequenceName,'extrinsics/'),'txt');
%extrinsicsC2W = permute(reshape(readValuesFromTxt(fullfile('http://sun3d.cs.princeton.edu/data/',sequenceName,'extrinsics',extrinsicsFiles(end).name)),4,3,[]),[2 1 3]);
[proj2D,z3] = project3dPtsTo2d(gridXYZ',data.Rtilt,[1,1],data.K);
proj2D = proj2D';
z3 = z3';
inCamera = z3 > 1.0 & z3 < cutoff & 0<proj2D(1,:) & proj2D(1,:)<imgSize(2) & 0<proj2D(2,:) & proj2D(2,:)<imgSize(1);
gridXYZ = gridXYZ(:,inCamera);





if opts.visualization
    
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

    %figure
    numCorners = size(gtCorner3D,2)/2;
    patch(gtCorner3D(1,1:numCorners),gtCorner3D(2,1:numCorners),gtCorner3D(3,1:numCorners),'g');
    hold on
    patch(gtCorner3D(1,numCorners+1:end),gtCorner3D(2,numCorners+1:end),gtCorner3D(3,numCorners+1:end),'g');
    hold on
    for i=1:numCorners-1
        patch(gtCorner3D(1,[i i+1 numCorners+i+1 numCorners+i]),gtCorner3D(2,[i i+1 numCorners+i+1 numCorners+i]),gtCorner3D(3,[i i+1 numCorners+i+1 numCorners+i]),'g');
        hold on
    end
    patch(gtCorner3D(1,[numCorners 1  numCorners+1 numCorners*2]),gtCorner3D(2,[numCorners 1  numCorners+1 numCorners*2]),gtCorner3D(3,[numCorners 1  numCorners+1 numCorners*2]),'g');
    axis equal
    alpha(0.6);    
% 
    
    hold on;
    plot3(gridXYZ(1,:),gridXYZ(2,:),gridXYZ(3,:),'.'); xlabel('x'); ylabel('y'); zlabel('z');
end

numCorners = size(gtCorner3D,2)/2;

%% inside the ground truth
[inon,on] = inpolygon(gridXYZ(1,:),gridXYZ(2,:),gtCorner3D(1,1:numCorners),gtCorner3D(2,1:numCorners));
in = inon&~on;
inGroundTruth = in & min(gtCorner3D(3,:)) < gridXYZ(3,:) & gridXYZ(3,:) < max(gtCorner3D(3,:));

intersectCount = zeros(1,length(inGroundTruth));
for i=find(inGroundTruth)    
    [xi,yi] = polyxpoly([0 gridXYZ(1,i)],[0 gridXYZ(2,i)],gtCorner3D(1,1:numCorners),gtCorner3D(2,1:numCorners));
    if ~isempty(xi)
        count = 0;
        for k=1:length(xi)
            disSq = xi(k)^2+yi(k)^2;
            if disSq > 0.5^2
                count=count+1;
            end
        end
        intersectCount(i) = count;
%         disp(count);
    end
end
inGroundTruth = inGroundTruth & (intersectCount==0);

if opts.visualization
    hold on;
    plot3(gridXYZ(1,inGroundTruth),gridXYZ(2,inGroundTruth),gridXYZ(3,inGroundTruth),'.g');
end

%% inside the prediction results
[inon,on] = inpolygon(gridXYZ(1,:),gridXYZ(2,:),outCorner3D(1,1:end/2),outCorner3D(2,1:end/2));
in = inon&~on;
inResult = in & min(outCorner3D(3,:)) < gridXYZ(3,:) & gridXYZ(3,:) < max(outCorner3D(3,:));

intersectCount = zeros(1,length(inResult));
for i=find(inResult)    
    %[xi,yi] = polyxpoly([0 gridXYZ(1,i)],[0 gridXYZ(3,i)],outCorner3D(1,1:end/2),outCorner3D(3,1:end/2));
    [xi,yi] = polyxpoly([0 gridXYZ(1,i)],[0 gridXYZ(2,i)],outCorner3D(1,1:end/2),outCorner3D(2,1:end/2));
    if ~isempty(xi)
        count = 0;
        for k=1:length(xi)
            disSq = xi(k)^2+yi(k)^2;
            if disSq > 0.5^2
                count=count+1;
            end
        end
        intersectCount(i) = count;
    end
end
inResult = inResult & (intersectCount==0);

if opts.visualization
    hold on;
    plot3(gridXYZ(1,inResult),gridXYZ(2,inResult),gridXYZ(3,inResult),'.k');
end

%% evaluate
intersectionGR = inGroundTruth & inResult;
unionGR = inGroundTruth | inResult;
iou = sum(intersectionGR) / sum(unionGR);

