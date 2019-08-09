function [cop, dir] = Unproject2DPoints(pt2, proj)
%cop 3x1
%dir nx3
%see multiview geometry Page 162

cop = -proj(:,1:3)\proj(:,4);
dir = (proj(:,1:3)\[pt2, ones(size(pt2,1),1)]')';