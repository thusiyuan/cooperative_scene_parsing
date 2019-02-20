function outCorner3D = ConvexBox(data, opts)
[rgb,points3d,imgZ]=read3dPoints(data);
XYZworld = double(points3d');
XYZworld = [[0;0;0] XYZworld];
XYZworld(:,sum(isnan(XYZworld))>0) =[];

zRange = prctile(XYZworld(3,:),[0.1 99.9]);
Khull = convhull(XYZworld(1,:),XYZworld(2,:));
outCorner3D(1,:) = XYZworld(1,Khull);
outCorner3D(2,:) = XYZworld(2,Khull);
outCorner3D(3,:) = zRange(1);

outCornerMax = outCorner3D;
outCornerMax(3,:) = zRange(2);
outCorner3D = [outCorner3D outCornerMax];

if opts.visualization
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

end
end