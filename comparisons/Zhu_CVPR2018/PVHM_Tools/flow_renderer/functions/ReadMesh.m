function mesh=readmesh(name)

if ~exist(name, 'file')
    fprintf('no such file!\n');
    mesh = [];
    return;
end

% maxnum=1000000;
% fid=fopen(name,'r');
% aline = fgetl(fid);
% vlist=zeros(maxnum,3);
% vcnt=0;
% vnlist=zeros(maxnum,3);
% vncnt=0;
% vtlist=zeros(maxnum,3);
% vtcnt = 0;
% flist=zeros(maxnum,3);
% fcnt=0;
% while ischar(aline),
%     if size(aline,2)>0
%         if aline(1)=='v' && aline(2)==' ',
%             v=sscanf(aline,'v %f %f %f');
%             vcnt=vcnt+1;
%             vlist(vcnt,:)=[v(1) v(2) v(3)];
%         end
%         
%         if aline(1)=='v' && aline(2)=='n' && aline(3)==' ',
%             vn=sscanf(aline,'vn %f %f %f');
%             vncnt=vncnt+1;
%             vnlist(vncnt,:)=[vn(1) vn(2) vn(3)];
%         end
%         
%         if aline(1)=='v' && aline(2)=='t' && aline(3)==' ',
%             vt=sscanf(aline,'vt %f %f %f');
%             vtcnt=vtcnt+1;
%             vtlist(vtcnt,:)=[vt(1) vt(2) vt(3)];
%         end        
% 
%         if aline(1)=='f'&&aline(2)==' ',
%             f=sscanf(aline,'f %d %d %d');
%             if size(f(:),1)==3
%                 fcnt=fcnt+1;
%                 flist(fcnt,:)=[f(1) f(2) f(3)];
%             else
%                 f = sscanf(aline, 'f %d/%d %d/%d %d/%d');
%                 if size(f(:),1)==6
%                     fcnt=fcnt+1;
%                     flist(fcnt,:)=[f(1) f(3) f(5)];
%                 else
%                     f = sscanf(aline, 'f %d/%d/%d %d/%d/%d %d/%d/%d');
%                     fcnt=fcnt+1;
%                     flist(fcnt,:)=[f(1) f(4) f(7)];                    
%                 end
%             end
%         end
%     end
%     aline = fgetl(fid);
% end
% fclose(fid);
% 
% vlist(vcnt+1:maxnum,:)=[];
% flist(fcnt+1:maxnum,:)=[];
% vnlist(vncnt+1:maxnum,:)=[];
% vtlist(vtcnt+1:maxnum,:)=[];

[~, ~, ext] = fileparts(name);

fullname = which(name);
if isempty(fullname)
    fullname = name;
end

if strcmp(ext, '.obj')
    [vlist, flist, vtlist, vnlist, vt_flist, vn_flist] = ReadMesh_c(fullname);
elseif  strcmp(ext, '.ply')
    [vlist, flist, vtlist, vnlist, ~, vt_flist, comments] = readply_c(fullname);
    vn_flist = [];
    if ~isempty(vt_flist)
        vtlist = reshape(vt_flist', 2, [])';
        vt_flist = reshape(1:size(flist,1)*3, 3, [])';
    end
end

mesh{1}=vlist;
mesh{2}=flist;
mesh{3}=vtlist;
mesh{4}=vnlist;
mesh{5}=vt_flist;
mesh{6}=vn_flist;
if isempty(mesh{6})
    mesh = mesh(1:5);   
    if isempty(mesh{5})
        mesh = mesh(1:4);
        if isempty(mesh{4})
            mesh = mesh(1:3);
            if isempty(mesh{3})
                mesh = mesh(1:2);
            end
        end
    end    
end