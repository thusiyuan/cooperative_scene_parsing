function extrinsicsC2W = getExtrinsicsC2W(sequenceName,opts)

extrinsicsFiles = dir(fullfile(opts.rootPath,sequenceName,'extrinsics/'));
fullfile(opts.rootPath,sequenceName,'extrinsics/');
extrinsicsC2W = permute(reshape(readValuesFromTxt(fullfile(opts.rootPath,sequenceName,'extrinsics',extrinsicsFiles(end).name)),4,3,[]),[2 1 3]);

end
%
