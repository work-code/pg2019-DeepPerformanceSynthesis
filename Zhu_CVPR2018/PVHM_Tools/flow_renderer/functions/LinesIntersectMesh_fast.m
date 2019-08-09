function [interPt, bary, lambda, faceId, texCoord] = LinesIntersectMesh_fast(linestart, linedir, mesh)

[interPt, bary, lambda, faceId] = LinesIntersectMesh_c(linestart, linedir, mesh{1}, mesh{2});

if max(size(mesh))>2 && ~isempty(mesh{3})
    dim = size(mesh{3},2);
    valid = faceId~=0;
    texCoord = zeros(length(faceId), dim);
    
    vt1 = mesh{3}(mesh{2}(faceId(valid),1), :);
    vt2 = mesh{3}(mesh{2}(faceId(valid),2), :);
    vt3 = mesh{3}(mesh{2}(faceId(valid),3), :);
    
    texCoord(valid,:) = vt1.*repmat(bary(valid,1), 1, dim)+vt2.*repmat(bary(valid,2), 1, dim)+vt3.*repmat(bary(valid,3), 1, dim);
else
    texCoord = [];
end






