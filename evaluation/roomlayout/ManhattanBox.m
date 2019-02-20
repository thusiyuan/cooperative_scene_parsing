function outCorner3D = ManhattanBox(data,opts)
[rgb,points3d,imgZ]=read3dPoints(data);
Space=struct('Rx',[nanmin(points3d(:,1)), nanmax(points3d(:,1))],...
             'Ry',[nanmin(points3d(:,2)),nanmax(points3d(:,2))],...
             'Rz',[nanmin(points3d(:,3)),nanmax(points3d(:,3))],'s',0.1);

imgSize= size(imread(data.depthpath));
[imageSeg,Rot]= getPlanSeg(points3d,Space,imgSize);

points3d_align = [[Rot(1:2,1:2)*points3d(:,[1,2])']', points3d(:,3)];

im = imread(data.rgbpath);

cx = data.K(1,3); cy = data.K(2,3);  
fx = data.K(1,1); fy = data.K(2,2); 

[imgX,imgY] = meshgrid(1:size(imgZ,2), 1:size(imgZ,1));   
imgX = (imgX-cx).*imgZ/fx;
imgY = (imgY-cy).*imgZ/fy;

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

% minIdx(imageSeg) finds 3 orthogonal directions?

if opts.visualization
    figure, imagesc(minIdx(imageSeg));
    cmap=[0.5 0.5 0.5; 1 0 0; 0 1 0; 0 0 1];
    colormap(cmap)
end

roi = minIdx(imageSeg)==1;
maxX = max(mhaX(roi(:)));
minX = min(mhaX(roi(:)));
if isempty(minX)
    minX = prctile(mhaX(:),0.5);
    maxX = prctile(mhaX(:),99.5);
else
    if minX > 0 || mean(mhaX(:)<minX)>0.08
        %minX = min(mhaX(:));
        minX = prctile(mhaX(:),0.5);
    end
    if maxX < 0 || mean(mhaX(:)>maxX)>0.08
        %maxX = max(mhaX(:));
        maxX = prctile(mhaX(:),99.5);
    end
end
minX = min(-1,minX);


roi = minIdx(imageSeg)==2;
maxY = max(mhaY(roi(:)));
minY = min(mhaY(roi(:)));
if isempty(minY)
    minY = prctile(mhaY(:),0.5);
    maxY = prctile(mhaY(:),99.5);
else
    if minY > 0 || mean(mhaY(:)<minY)>0.08
        %minY = min(mhaY(:));
        minY = prctile(mhaY(:),0.5);
    end
    if maxY < 0 || mean(mhaY(:)>maxY)>0.08
        %maxY = max(mhaY(:));
        maxY = prctile(mhaY(:),99.5);
    end
end
minY = min(-1,minY);

roi = minIdx(imageSeg)==3;
maxZ = max(mhaZ(roi(:)));
minZ = min(mhaZ(roi(:)));

if isempty(minZ)
    minZ = prctile(mhaZ(:),0.5);
    maxZ = prctile(mhaZ(:),99.5);    
end

minZ = min(0,minZ);

if maxZ-minZ>=2.8-0.1 % yes there is a ceiling!
    ceiling = [...
    minX minY maxZ
    maxX minY maxZ
    maxX maxY maxZ
    minX maxY maxZ]';
else
    ceiling = [];
    maxZ = 4;
end

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



%% predicted room
outCorner3D = [floor floor];
outCorner3D(3,(size(floor,2)+1):end) = max(wallX(3,:));
outCorner3D(1:2,:) = Rot(1:2,1:2)'*outCorner3D(1:2,:);

%% cordinate transformation 
%extrinsicsC2W = getExtrinsicsC2W(sequenceName, opts);
% outCorner3D = data.Rtilt' * outCorner3D;
% outCorner3D(3,:) = - outCorner3D(3,:);
% outCorner3D([2 3],:) = outCorner3D([3 2],:);
% outCorner3D = extrinsicsC2W(1:3,1:3) * outCorner3D;




%% visualization of the result
if opts.visualization
    figure
    clf
    vis_point_cloud(points3d_align,rgb,10,10000); 
    xlabel('x'); ylabel('y'); zlabel('z');
    hold on; patch(wallX(1,:),wallX(2,:),wallX(3,:),'r')
    hold on; patch(wallY(1,:),wallY(2,:),wallY(3,:),'g')
    hold on; patch(floor(1,:),floor(2,:),floor(3,:),'b')
    if ~isempty(ceiling)
        hold on; patch(ceiling(1,:),ceiling(2,:),ceiling(3,:),'k');
    end
    alpha(0.6);

    % 2D visualization
    figure, imshow(im);

    %{
    Xdraw = floor;
    Xdraw(1:2,:) = Rot(1:2,1:2)'*Xdraw(1:2,:);
    Xdraw = data.Rtilt' * Xdraw;
    Xdraw(3,:) = - Xdraw(3,:);
    Xdraw([2 3],:) = Xdraw([3 2],:);
    xdraw = data.K * Xdraw;
    xdraw = xdraw(1:2,:)./xdraw([3 3],:);
    %}
    xdraw = project3Dto2D(wallX,Rot,data);
    hold on; plot(xdraw(1,[1:end 1]),xdraw(2,[1:end 1]),'-r','LineWidth',5);
    xdraw = project3Dto2D(wallY,Rot,data);
    hold on; plot(xdraw(1,[1:end 1]),xdraw(2,[1:end 1]),'-g','LineWidth',4);
    xdraw = project3Dto2D(floor,Rot,data);
    hold on; plot(xdraw(1,[1:end 1]),xdraw(2,[1:end 1]),'-b','LineWidth',3);
    title('Output of Algorithms');

    figure
    vis_point_cloud(points3d,rgb,10,10000); 
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
   
    
    %{
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
    %}

end

