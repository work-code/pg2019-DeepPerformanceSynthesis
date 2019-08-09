function [mesh, vcolor, texturename] = readply(name)
%[mesh, vcolor, texturename] = readply(name)
%comments is used to define textures

if ~exist(name, 'file')
    fprintf('no such file!\n');
    mesh = [];
    return;
end

fullname = which(name);
if isempty(fullname)
    fullname = name;
end


[vlist, flist, vtlist, vnlist, vcolor, vt_flist, comments] = readply_c(fullname);

if ~isempty(vt_flist) && isempty(vtlist)
    vt_flist = [vt_flist(:,1:2);vt_flist(:,3:4);vt_flist(:,5:6)];
    [vtlist, ~, vt_flist] = unique(vt_flist, 'rows');
    vt_flist = reshape(vt_flist, [], 3);
end

mesh{1}=vlist;
mesh{2}=flist;
mesh{3}=vtlist;
mesh{4}=vnlist;
mesh{5}=vt_flist;

if isempty(mesh{5})
    mesh = mesh(1:4);
    if isempty(mesh{4})
        mesh = mesh(1:3);
        if isempty(mesh{3})
            mesh = mesh(1:2);
        end
    end
end
texturename = [];
if ~isempty(comments)
    for i=1:length(comments)
        pat = 'TextureFile (?<texturename>.+)';
        nameline = regexp(comments{i}, pat, 'names');
        if ~isempty(nameline)
            texturename = nameline.texturename;
            break;
        end
    end
end