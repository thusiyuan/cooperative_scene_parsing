function [rgb,points3d,depthInpaint,imsize]=read3dPoints(data)
         depthVis = imread(data.depthpath);
         imsize = size(depthVis);
         depthInpaint = bitor(bitshift(depthVis,-3), bitshift(depthVis,16-3));
         depthInpaint = single(depthInpaint)/1000; 
         depthInpaint(depthInpaint >10 )=10;
         [rgb,points3d]=read_3d_pts_general(depthInpaint,data.K,size(depthInpaint),data.rgbpath,data.crop);
         points3d = (data.Rtilt*points3d')';
end