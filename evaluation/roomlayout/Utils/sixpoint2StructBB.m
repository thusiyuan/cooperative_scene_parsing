function bb3d = sixpoint2StructBB(box,name, label,conf)
basis2d(:,2) = box(1:2,2)-box(1:2,6);
basis2d(:,1) = box(1:2,7)-box(1:2,6);
basis2d(:,1) = basis2d(:,1)/norm(basis2d(:,1));
basis2d(:,2) = basis2d(:,2)/norm(basis2d(:,2));
centroid = .5*(box(:,1)+box(:,7));
coeffsy = 0.5*norm(box(:,2)-box(:,6));
coeffsx = 0.5*norm(box(:,7)-box(:,6));
coeffsz = 0.5*norm(box(:,2)-box(:,1));

bb3d = create_bounding_box_3d(basis2d, centroid, [coeffsx,coeffsy,coeffsz]);
bb3d.className = name;
bb3d.confidence = conf;
bb3d.classid = label;
end