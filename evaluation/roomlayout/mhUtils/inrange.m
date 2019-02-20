function inRangeIX = inrange(X,R)
leftBound = R(:,1);
rightBound = R(:,2);
if any(leftBound > rightBound),
    error('Rows of RANGE must have form [LOW HIGH].')
end

inRangeIX = sum(bsxfun(@lt,-1*X(:),-1*leftBound(:)') & bsxfun(@lt,X(:),rightBound(:)'),2);
