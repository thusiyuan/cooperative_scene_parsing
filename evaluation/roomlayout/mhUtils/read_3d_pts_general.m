function [rgb,points3d,points3dMatrix]=read_3d_pts_general(depthInpaint,K,depthInpaintsize,imageName,crop)
    %K is [fx 0 cx; 0 fy cy; 0 0 1];  
    %K = frames.K;
    if ~isempty(K)
        cx = K(1,3); cy = K(2,3);  
        fx = K(1,1); fy = K(2,2); 
    else
        fx = 5.1885790117450188e+02;
        fy = 5.1946961112127485e+02;
        cx = 3.2558244941119034e+02;
        cy = 2.5373616633400465e+02;
    end
    invalid = depthInpaint==0;
    if ~isempty(imageName)
        im = imread(imageName);
        rgb = im2double(im);  
    else
        rgb =double(cat(3,zeros(depthInpaintsize(1),depthInpaintsize(2)),...
                    ones(depthInpaintsize(1),depthInpaintsize(2)),...
                    zeros(depthInpaintsize(1),depthInpaintsize(2))));
    end
    rgb = reshape(rgb, [], 3);
    %3D points
    [x,y] = meshgrid(crop(2)-1+(1:depthInpaintsize(2)), crop(1)-1+(1:depthInpaintsize(1)));   
    x3 = (x-cx).*depthInpaint*1/fx;  
    y3 = (y-cy).*depthInpaint*1/fy;  
    z3 = depthInpaint;  
    points3dMatrix =cat(3,x3,z3,-y3);
    points3dMatrix(cat(3,invalid,invalid,invalid))=NaN;
    points3d = [x3(:) z3(:) -y3(:)];
    points3d(invalid(:),:) =NaN;
end