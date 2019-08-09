% Spin the model according to z axis.
% Direction: from y axis to axis.
% theta is degree (0-360)
function [MeshOut] = rotate_z_axis(MeshIn, theta)

MeshOut = zeros(size(MeshIn));
MeshOut(:,1) = MeshIn(:,1)*cos(theta) + MeshIn(:,2)*sin(theta);
MeshOut(:,2) = MeshIn(:,2)*cos(theta) - MeshIn(:,1)*sin(theta);
MeshOut(:,3) = MeshIn(:,3);

