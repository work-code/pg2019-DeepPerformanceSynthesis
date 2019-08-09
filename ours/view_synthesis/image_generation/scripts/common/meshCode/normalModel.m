function [shape,center] = normalModel(shape,scale)
centerX = mean(shape.X(:,1));
centerY = mean(shape.Y(:,1));
centerZ = mean(shape.Z(:,1));
center = [centerX, centerY, centerZ];

shape.X = shape.X(:,1)-repmat(centerX,shape.nv,1);
shape.Y = shape.Y(:,1)-repmat(centerY,shape.nv,1);
shape.Z = shape.Z(:,1)-repmat(centerZ,shape.nv,1);


shape.X = shape.X(:,1).*scale;
shape.Y = shape.Y(:,1).*scale;
shape.Z = shape.Z(:,1).*scale;


end