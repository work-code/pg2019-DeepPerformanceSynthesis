function uv = Project3DPoints(pt3, proj)
%pt3: n*3

v = size(pt3,1);
if size(proj,2)==4
    uv = (proj*[pt3, ones(v,1)]')';
elseif size(proj,2)==3  %3*3 mat, i.e, KK
    uv = pt3*proj';
end

uv = [uv(:,1)./uv(:,3), uv(:,2)./uv(:,3)];

