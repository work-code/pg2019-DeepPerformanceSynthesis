function [depth_map]=DepthMap_front(MeshFolderPath,OutputMeshPath,Pose1_name,Pose1_name_ply,K,Rt0)

% modified by Hao Zhu 2017-8-25
% compute normal map

%% convert blender .obj to meshlab .ply
disp( [ '"' fullfile(MeshFolderPath, Pose1_name) '"'])
disp( [ '"' fullfile(OutputMeshPath, Pose1_name_ply) '"'])
if ~exist(fullfile(OutputMeshPath, Pose1_name_ply),'file')
 [t,t1] = system(sprintf('C:\\PROGRA~1\\VCG\\MeshLab\\meshlabserver.exe -i %s -o %s', [ '"' fullfile(MeshFolderPath, Pose1_name) '"'], [ '"' fullfile(OutputMeshPath, Pose1_name_ply) '"']));
end

%% read in two meshes 
mesh_pose1=readply(fullfile(OutputMeshPath, Pose1_name_ply));

%% convert the [X Y Z] to [X -Z Y]
mesh_pose1{1}=[mesh_pose1{1}(:,1) -mesh_pose1{1}(:,3) mesh_pose1{1}(:,2)];

% Check visibility of 

uv = Project3DPoints(mesh_pose1{1}, K*Rt0);
uv = unique(uv, 'rows');

dt = delaunayTriangulation(uv(:,1),uv(:,2));
k = convexHull(dt);

mask = poly2mask(uv(k,1), uv(k,2),200, 200);

[v_all, u_all] = find(mask);

[center, dir] = Unproject2DPoints([u_all v_all], K*Rt0);
[~, dir0] = Unproject2DPoints([100.5 100.5], K*Rt0);
dir0 = dir0/norm(dir0);

v_all_num = length(v_all);

[interPtOnPose1, bary, lambda, faceId] = LinesIntersectMesh_fast(repmat(center(:)', v_all_num, 1), dir, mesh_pose1);
%mask(mask) = faceId~=0;


% find correspondence 3D points on Tpose
valid=faceId~=0;
interPtOnPose1=interPtOnPose1(valid,:);

size_tmp = size(interPtOnPose1);
size_tmp = size_tmp(1);
depth_list = zeros(size_tmp(1),1);
% compute normal
for i = 1:size_tmp
    depth_list(i) = dot((interPtOnPose1(i,:) - center'), dir0);

uv_pose1 = Project3DPoints(interPtOnPose1, K*Rt0);

index=sub2ind([200 200],floor(uv_pose1(:,2)+0.5),floor(uv_pose1(:,1)+0.5));

depth_map=zeros(200,200);
depth_map(round(index))=depth_list(:);

% im=imread('testtest.png');
% im=im2double(rgb2gray(im));
% [warpI2,I]=warpFlow(im,flow_x,flow_y);

%% test 
% test=zeros(224,224);
% new_u=round(uv_pose1(:,2)+flow_y(index));
% new_v=round(uv_pose1(:,1)+flow_x(index));
% index_new=sub2ind([224 224],new_u,new_v);
% 
% test(index_new)=1;
% figure;imshow(test)
end