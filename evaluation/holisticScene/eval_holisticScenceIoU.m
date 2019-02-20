function iou = eval_holisticScenceIoU(data,gtRoom3D,gtObjectStruct,outRoom3D,outObjectStruct,visualization)
cutoff = 5.5;
delta  = 0.1;
if ~exist('visualization','var')
    visualization = false;
end

gtObject3D.objects = converBB(gtObjectStruct);
predictObject3D.objects = converBB(outObjectStruct);

%% evaluate the accuracy
rangeX = min(min(outRoom3D(1,:)),min(gtRoom3D(1,:)))-delta:delta:max(max(outRoom3D(1,:)),max(gtRoom3D(1,:)))+delta;
rangeY = min(min(outRoom3D(2,:)),min(gtRoom3D(2,:)))-delta:delta:max(max(outRoom3D(2,:)),max(gtRoom3D(2,:)))+delta;
rangeZ = min(min(outRoom3D(3,:)),min(gtRoom3D(3,:)))-delta:delta:max(max(outRoom3D(3,:)),max(gtRoom3D(3,:)))+delta;
[gridX, gridY, gridZ] = ndgrid(rangeX,rangeY,rangeZ);
gridXYZ = [gridX(:)'; gridY(:)'; gridZ(:)'];
clear gridX gridY gridZ
imgSize= size(imread(data.depthpath));

[proj2D,z3] = project3dPtsTo2d(gridXYZ',data.Rtilt,[1,1],data.K);
proj2D = proj2D';
z3 = z3';
inCamera = z3 > 1.0 & z3 < cutoff & 0<proj2D(1,:) & proj2D(1,:)<imgSize(2) & 0<proj2D(2,:) & proj2D(2,:)<imgSize(1);
gridXYZ = gridXYZ(:,inCamera);


numCorners = size(gtRoom3D,2)/2;
[inon,on] = inpolygon(gridXYZ(1,:),gridXYZ(2,:),gtRoom3D(1,1:numCorners),gtRoom3D(2,1:numCorners));
in = inon&~on;
inGroundTruth = in & min(gtRoom3D(3,:)) < gridXYZ(3,:) & gridXYZ(3,:) < max(gtRoom3D(3,:));

intersectCount = zeros(1,length(inGroundTruth));
for i=find(inGroundTruth)    
    [xi,yi] = polyxpoly([0 gridXYZ(1,i)],[0 gridXYZ(2,i)],gtRoom3D(1,1:numCorners),gtRoom3D(2,1:numCorners));
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
        [inon, on] = inpolygon(gridXYZ(1,:),gridXYZ(2,:), gtObject3D.objects{j}(1,:), gtObject3D.objects{j}(2,:));
        in = inon&~on;
        in = in & min(gtObject3D.objects{j}(3,:)) < gridXYZ(3,:) & gridXYZ(3,:) < max(gtObject3D.objects{j}(3,:));
        inGroundTruth = inGroundTruth & ~in;
    end
end


%% inside the prediction results
[inon,on] = inpolygon(gridXYZ(1,:),gridXYZ(2,:),outRoom3D(1,1:end/2),outRoom3D(2,1:end/2));
in = inon&~on;
inResult = in & min(outRoom3D(3,:)) < gridXYZ(3,:) & gridXYZ(3,:) < max(outRoom3D(3,:));

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
for j=1:length(predictObject3D.objects)
    if ~isempty(predictObject3D.objects{j})
        [inon, on] = inpolygon(gridXYZ(1,:),gridXYZ(2,:), predictObject3D.objects{j}(1,:), predictObject3D.objects{j}(2,:));
        in = inon&~on;
        in = in & min(predictObject3D.objects{j}(3,:)) < gridXYZ(3,:) & gridXYZ(3,:) < max(predictObject3D.objects{j}(3,:));
        inResult = inResult & ~in;
    end
end


%% evaluate
intersectionGR = inGroundTruth & inResult;
unionGR = inGroundTruth | inResult;
iou = sum(intersectionGR) / sum(unionGR);
%% visualization 
if visualization
    order = [1,2,3,4,1,5,6,2,3,7,6,7,8,4,8,5];
    %% draw gt 
    figure
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
        plot3(gtObject3D.objects{j}(1,order), gtObject3D.objects{j}(2,order),  gtObject3D.objects{j}(3,order),'-k','LineWidth',2);
    end
   
    %% draw prediction 
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
    for j=1:length(predictObject3D.objects)
        plot3(predictObject3D.objects{j}(1,order), predictObject3D.objects{j}(2,order),  predictObject3D.objects{j}(3,order),'-b','LineWidth',2);
    end
    hold on;
%     campos([0, 0, 0]);
%     camup([0, 0, 1]);
    %
    hold on;
    plot3(gridXYZ(1,inResult),gridXYZ(2,inResult),gridXYZ(3,inResult),'.k');
    %%
    hold on;
    plot3(gridXYZ(1,inGroundTruth),gridXYZ(2,inGroundTruth),gridXYZ(3,inGroundTruth),'.b');
    
end

end
function objects= converBB(bb3ds)
if ~isempty(bb3ds)
    for bbidx=1:length(bb3ds)
        corners = get_corners_of_bb3d(bb3ds(bbidx));
%         corners = [data.Rtilt'*corners']';
%         corners = [[1 0 0; 0 0 -1 ;0 1 0]*corners']';
%         corners = [data.anno_extrinsics*corners']';
%         X =corners(1:4,1)';
%         Z =corners(1:4,3)';
%         Ymin = min(corners(1,2),corners(5,2));
%         Ymax = max(corners(1,2),corners(5,2));
%         rectangle =true;
%         P =struct('Ymin',Ymin,'Ymax',Ymax,'rectangle',rectangle,'X',X,'Z',Z);
%         polygon=cell(1,1);
%         polygon{1} =P;
%         object.polygon = polygon;
        %object =struct('name',bb3ds(bbidx).classname,'polygon',polygon);
        objects{bbidx} =corners';
    end
else
    objects ={};
end
end

function [points2d,z3] = project3dPtsTo2d(points3d,Rtilt,crop,K)
    %% inverse of get_aligned_point_cloud
    points3d =[Rtilt'*points3d']';
    
    %% inverse rgb_plane2rgb_world
    
    
    % Now, swap Y and Z.
    points3d(:, [2, 3]) = points3d(:,[3, 2]);
    
    % Make the original consistent with the camera location:
    x3 = points3d(:,1);
    y3 = -points3d(:,2);
    z3 = points3d(:,3);
    
    xx = x3 * K(1,1) ./ z3 + K(1,3);
    yy = y3 * K(2,2) ./ z3 + K(2,3);
    

    xx = xx - crop(2) + 1;
    yy = yy - crop(1) + 1;

    points2d = [xx yy];
end