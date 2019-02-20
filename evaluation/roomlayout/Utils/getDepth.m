function depth = getDepth(sequenceName, opts)
depth = imread(fullfile(opts.rootPath,sequenceName,'depth/',depthFiles(end).name));
depth = bitor(bitshift(depth,-3), bitshift(depth,16-3));
depth = single(depth)/1000;
e
end
