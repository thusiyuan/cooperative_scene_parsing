function xdraw = project3Dto2D(Xdraw,Rot,data, swap)

Xdraw(1:2,:) = Rot(1:2,1:2)'*Xdraw(1:2,:);
Xdraw = data.Rtilt' * Xdraw;
if ~exist('swap','var') || (exist('swap','var') && swap)
    Xdraw(3,:) = - Xdraw(3,:);
    Xdraw([2 3],:) = Xdraw([3 2],:);
end
xdraw = data.K * Xdraw;
xdraw = xdraw(1:2,:)./xdraw([3 3],:);