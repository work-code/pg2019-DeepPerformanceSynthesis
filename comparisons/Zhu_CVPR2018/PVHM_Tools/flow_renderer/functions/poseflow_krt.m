function [flow_x, flow_y, uv_src_mask]=poseflow_krt(MeshFolderPath,OutputMeshPath,mesh_obj_filename,mesh_ply_filename,K_src,Rt_src,K_tgt,Rt_tgt)




%% convert blender .obj to meshlab .ply

if ~exist(fullfile(OutputMeshPath, mesh_ply_filename),'file');
 [~,~] = system(sprintf('meshlabserver.exe -i %s -o %s', [ '"' fullfile(MeshFolderPath, mesh_obj_filename) '"'], [ '"' fullfile(OutputMeshPath, mesh_ply_filename) '"']));
end

%% read in two meshes 
mesh=readply(fullfile(OutputMeshPath, mesh_ply_filename));

%% convert the [X Y Z] to [X -Z Y]
mesh{1}=[mesh{1}(:,1) -mesh{1}(:,3) mesh{1}(:,2)];

% Check visibility of 

uv = Project3DPoints(mesh{1}, K_tgt*Rt_tgt);
uv = unique(uv, 'rows');

dt = delaunayTriangulation(uv(:,1),uv(:,2));
k = convexHull(dt);

mask = poly2mask(uv(k,1), uv(k,2),200, 200);

[v_all, u_all] = find(mask);

[center, dir] = Unproject2DPoints([u_all v_all], K_tgt*Rt_tgt);
v_all_num = length(v_all);

[interPtOnPose1, bary, lambda, faceId] = LinesIntersectMesh_fast(repmat(center(:)', v_all_num, 1), dir, mesh);
%mask(mask) = faceId~=0;


% find correspondence 3D points on Tpose


valid=faceId~=0;
interPt=interPtOnPose1(valid,:);


%v1=mesh{1}(mesh{2}(faceId(valid),1), :);
%v2=mesh{1}(mesh{2}(faceId(valid),2), :);
%v3=mesh{1}(mesh{2}(faceId(valid),3), :);
%interPtonTpose = v1.*repmat(bary(valid,1), 1, 3)+v2.*repmat(bary(valid,2), 1, 3)+v3.*repmat(bary(valid,3), 1, 3);

% compute flow 

uv_view_src = Project3DPoints(interPt, K_src*Rt_src);
uv_view_tgt = Project3DPoints(interPt, K_tgt*Rt_tgt);

% index_src=sub2ind([200 200],floor(uv_view_src(:,2)+0.5),floor(uv_view_src(:,1)+0.5));

index_tgt=sub2ind([200 200],floor(uv_view_tgt(:,2)+0.5),floor(uv_view_tgt(:,1)+0.5));

%uv_diff=uv_Tpose-uv_pose1;
uv_diff = uv_view_src - uv_view_tgt;

flow_x=zeros(200,200);
flow_y=zeros(200,200);

%[sub_ind_u,  sub_ind_v]=ind2sub([200 200],index);

uv_src_mask=zeros(200,200);
%uv_tgt_mask=zeros(200,200);
 
% for test
uv_src_mask(round(index_tgt))=1;
%uv_tgt_mask(round(index_tgt))=1;
 
flow_x(round(index_tgt))=uv_diff(:,1);
flow_y(round(index_tgt))=uv_diff(:,2);


end