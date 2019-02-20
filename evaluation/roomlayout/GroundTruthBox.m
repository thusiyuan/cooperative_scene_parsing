function gtCorner3D = GroundTruthBox(sequenceName,opts)

groundTruthName = [opts.rootPath sequenceName '/annotation3Dlayout/index.json'];

%%

json=loadjson(groundTruthName);
for objectID=1:length(json.objects)
    try
        groundTruth = json.objects{objectID}.polygon{1};
        break;
    catch
    end
end
numCorners = length(groundTruth.X);
gtCorner3D(1,:) = [groundTruth.X groundTruth.X];
gtCorner3D(2,:) = [repmat(groundTruth.Ymin,[1 numCorners]) repmat(groundTruth.Ymax,[1 numCorners])];
gtCorner3D(3,:) = [groundTruth.Z groundTruth.Z];

if opts.visualization
    
    data=readframe(fullfile(opts.rootPath,sequenceName));
    [rgb,points3d,imgZ]=read3dPoints(data);
    im = imread(data.rgbpath);

    extrinsicsC2W = getExtrinsicsC2W(sequenceName,opts);
    
    xdraw = data.K * extrinsicsC2W(1:3,1:3)' * gtCorner3D;
    xdraw = xdraw(1:2,:)./xdraw([3 3],:);

    figure, imshow(im);
    hold on; plot(xdraw(1,[1:numCorners 1]),xdraw(2,[1:numCorners 1]),'-k','LineWidth',3);
    hold on; plot(xdraw(1,numCorners+[1:numCorners 1]),xdraw(2,numCorners+[1:numCorners 1]),'-b','LineWidth',3);
    for cornerID=1:numCorners
        hold on; plot(xdraw(1,[cornerID numCorners+cornerID]),xdraw(2,[cornerID numCorners+cornerID]),'-g','LineWidth',3);
    end
    title('Ground Truth');


    figure
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
    title('Ground Truth');


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
