function iou = evaluateRoomObject3D(sequenceName,gtRoom3D,gtObject3D,outRoom3D,outObject3D,visualization,cutoff,delta)

%% set parameters

% set cutoff distance
if ~exist('cutoff','var')
    cutoff = 5.5;
end

% granuality
if ~exist('delta','var')
    delta  = 0.1;
end

if ~exist('visualization','var')
    visualization = false;
end


%% evaluate the accuracy
rangeX = min(min(outRoom3D(1,:)),min(gtRoom3D(1,:)))-delta:delta:max(max(outRoom3D(1,:)),max(gtRoom3D(1,:)))+delta;
rangeY = min(min(outRoom3D(2,:)),min(gtRoom3D(2,:)))-delta:delta:max(max(outRoom3D(2,:)),max(gtRoom3D(2,:)))+delta;
rangeZ = min(min(outRoom3D(3,:)),min(gtRoom3D(3,:)))-delta:delta:max(max(outRoom3D(3,:)),max(gtRoom3D(3,:)))+delta;
[gridX, gridY, gridZ] = ndgrid(rangeX,rangeY,rangeZ);
gridXYZ = [gridX(:)'; gridY(:)'; gridZ(:)'];
clear gridX gridY gridZ
%plot3(gridXYZ(1,:),gridXYZ(2,:),gridXYZ(3,:),'.');

data=readframe(fullfile('/n/fs/sun3d/data/',sequenceName));
imgSize= size(imread(data.depthpath));

extrinsicsFiles = dirSmart(fullfile('http://sun3d.cs.princeton.edu/data/',sequenceName,'extrinsics/'),'txt');
extrinsicsC2W = permute(reshape(readValuesFromTxt(fullfile('http://sun3d.cs.princeton.edu/data/',sequenceName,'extrinsics',extrinsicsFiles(end).name)),4,3,[]),[2 1 3]);


proj2D = data.K * extrinsicsC2W(1:3,1:3)' * gridXYZ;
proj2D(1:2,:) = proj2D(1:2,:) ./ proj2D([3 3],:);
inCamera = proj2D(3,:) > 1.0 & proj2D(3,:)< cutoff & 0<proj2D(1,:) & proj2D(1,:)<imgSize(2) & 0<proj2D(2,:) & proj2D(2,:)<imgSize(1);
gridXYZ = gridXYZ(:,inCamera);





if visualization
    
    figure

    
    %alpha(0.6);
    %title('Prediction Result');

    %figure
    numCorners = size(gtRoom3D,2)/2;
    patch(gtRoom3D(1,1:numCorners),gtRoom3D(2,1:numCorners),gtRoom3D(3,1:numCorners),'g');
    hold on
    patch(gtRoom3D(1,numCorners+1:end),gtRoom3D(2,numCorners+1:end),gtRoom3D(3,numCorners+1:end),'g');
    hold on
    for i=1:numCorners-1
        patch(gtRoom3D(1,[i i+1 numCorners+i+1 numCorners+i]),gtRoom3D(2,[i i+1 numCorners+i+1 numCorners+i]),gtRoom3D(3,[i i+1 numCorners+i+1 numCorners+i]),'g');
        hold on
    end
    patch(gtRoom3D(1,[numCorners 1  numCorners+1 numCorners*2]),gtRoom3D(2,[numCorners 1  numCorners+1 numCorners*2]),gtRoom3D(3,[numCorners 1  numCorners+1 numCorners*2]),'g');
    axis equal
       

    for j=1:length(gtObject3D.objects)
        if ~isempty(gtObject3D.objects{j})
            plot3(gtObject3D.objects{j}.polygon{1}.X([1:end 1]), repmat(gtObject3D.objects{j}.polygon{1}.Ymin,1,5), gtObject3D.objects{j}.polygon{1}.Z([1:end 1]),'-k','LineWidth',2);
            plot3(gtObject3D.objects{j}.polygon{1}.X([1:end 1]), repmat(gtObject3D.objects{j}.polygon{1}.Ymax,1,5), gtObject3D.objects{j}.polygon{1}.Z([1:end 1]),'-k','LineWidth',2);
            for k=1:4
                plot3(gtObject3D.objects{j}.polygon{1}.X([k k]),[gtObject3D.objects{j}.polygon{1}.Ymin,gtObject3D.objects{j}.polygon{1}.Ymax], gtObject3D.objects{j}.polygon{1}.Z([k k]),'-k','LineWidth',2);
            end
        end
    end
    xlabel('x');
    ylabel('y');
    zlabel('z');
    
    
    numCorners = size(outRoom3D,2)/2;
    patch(outRoom3D(1,1:numCorners),outRoom3D(2,1:numCorners),outRoom3D(3,1:numCorners),'k');
    hold on
    patch(outRoom3D(1,numCorners+1:end),outRoom3D(2,numCorners+1:end),outRoom3D(3,numCorners+1:end),'b');
    hold on
    for i=1:numCorners-1
        patch(outRoom3D(1,[i i+1 numCorners+i+1 numCorners+i]),outRoom3D(2,[i i+1 numCorners+i+1 numCorners+i]),outRoom3D(3,[i i+1 numCorners+i+1 numCorners+i]),'r');
        hold on
    end
    patch(outRoom3D(1,[numCorners 1  numCorners+1 numCorners*2]),outRoom3D(2,[numCorners 1  numCorners+1 numCorners*2]),outRoom3D(3,[numCorners 1  numCorners+1 numCorners*2]),'r');
    axis equal
    
    alpha(0.01); 
    
    for j=1:length(outObject3D.objects)
        if ~isempty(outObject3D.objects{j})
            plot3(outObject3D.objects{j}.polygon{1}.X([1:end 1]), repmat(outObject3D.objects{j}.polygon{1}.Ymin,1,5), outObject3D.objects{j}.polygon{1}.Z([1:end 1]),'-b','LineWidth',2);
            plot3(outObject3D.objects{j}.polygon{1}.X([1:end 1]), repmat(outObject3D.objects{j}.polygon{1}.Ymax,1,5), outObject3D.objects{j}.polygon{1}.Z([1:end 1]),'-b','LineWidth',2);
            for k=1:4
                plot3(outObject3D.objects{j}.polygon{1}.X([k k]),[outObject3D.objects{j}.polygon{1}.Ymin,outObject3D.objects{j}.polygon{1}.Ymax], outObject3D.objects{j}.polygon{1}.Z([k k]),'-b','LineWidth',2);
            end
        end
    end
    xlabel('x');
    ylabel('y');
    zlabel('z');
    
        
    
    
    %hold on;
    %plot3(gridXYZ(1,:),gridXYZ(2,:),gridXYZ(3,:),'.'); xlabel('x'); ylabel('y'); zlabel('z');
end

numCorners = size(gtRoom3D,2)/2;

%% inside the ground truth
[inon,on] = inpolygon(gridXYZ(1,:),gridXYZ(3,:),gtRoom3D(1,1:numCorners),gtRoom3D(3,1:numCorners));
in = inon&~on;
inGroundTruth = in & min(gtRoom3D(2,:)) < gridXYZ(2,:) & gridXYZ(2,:) < max(gtRoom3D(2,:));

intersectCount = zeros(1,length(inGroundTruth));
for i=find(inGroundTruth)    
    [xi,yi] = polyxpoly([0 gridXYZ(1,i)],[0 gridXYZ(3,i)],gtRoom3D(1,1:numCorners),gtRoom3D(3,1:numCorners));
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
inGroundTruth = inGroundTruth & (intersectCount==0);


for j=1:length(gtObject3D.objects)
    if ~isempty(gtObject3D.objects{j})
        [inon, on] = inpolygon(gridXYZ(1,:),gridXYZ(3,:), gtObject3D.objects{j}.polygon{1}.X, gtObject3D.objects{j}.polygon{1}.Z);
        in = inon&~on;
        in = in & gtObject3D.objects{j}.polygon{1}.Ymin < gridXYZ(2,:) & gridXYZ(2,:) < gtObject3D.objects{j}.polygon{1}.Ymax;
        inGroundTruth = inGroundTruth & ~in;
    end
end


if visualization
    hold on;
    plot3(gridXYZ(1,inGroundTruth),gridXYZ(2,inGroundTruth),gridXYZ(3,inGroundTruth),'.b');
end

%% inside the prediction results
[inon,on] = inpolygon(gridXYZ(1,:),gridXYZ(3,:),outRoom3D(1,1:end/2),outRoom3D(3,1:end/2));
in = inon&~on;
inResult = in & min(outRoom3D(2,:)) < gridXYZ(2,:) & gridXYZ(2,:) < max(outRoom3D(2,:));

intersectCount = zeros(1,length(inResult));
for i=find(inResult)    
    [xi,yi] = polyxpoly([0 gridXYZ(1,i)],[0 gridXYZ(3,i)],outRoom3D(1,1:end/2),outRoom3D(3,1:end/2));
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


for j=1:length(outObject3D.objects)
    if ~isempty(outObject3D.objects{j})
        [inon, on] = inpolygon(gridXYZ(1,:),gridXYZ(3,:), outObject3D.objects{j}.polygon{1}.X, outObject3D.objects{j}.polygon{1}.Z);
        in = inon&~on;
        in = in & outObject3D.objects{j}.polygon{1}.Ymin < gridXYZ(2,:) & gridXYZ(2,:) < outObject3D.objects{j}.polygon{1}.Ymax;
        inResult = inResult & ~in;
    end
end


if visualization
    hold on;
    plot3(gridXYZ(1,inResult),gridXYZ(2,inResult),gridXYZ(3,inResult),'.k');
end

%% evaluate
intersectionGR = inGroundTruth & inResult;
unionGR = inGroundTruth | inResult;
iou = sum(intersectionGR) / sum(unionGR);

