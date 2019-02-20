function N = compute3dNormal(points,rgb)
    if size(points,2) ~= 3, error('wrong input format. Points should be an Nx3 matrix.'); end
    
    imgSize = [471-45+1,601-41+1];
    X=reshape(points(:,1),imgSize);
    Y=reshape(points(:,2),imgSize);
    Z=reshape(points(:,3),imgSize);
    [Nx,Ny,Nz]=surfnorm(X,-Z,Y);
    
     mask = zeros(imgSize);
     mask(50:250,100:200) = 1;
     idx = 1:10:prod(imgSize);%logical(mask(:));
     figure,
     scatter3(X(idx),-Z(idx),Y(idx),10*ones(length(X(idx)),1),rgb(idx,:),'filled');
     axis equal; hold on;
     quiver3(X(idx),-Z(idx),Y(idx),Nx(idx),Ny(idx),Nz(idx));
    
    N = [Nx -Nz Ny];
end