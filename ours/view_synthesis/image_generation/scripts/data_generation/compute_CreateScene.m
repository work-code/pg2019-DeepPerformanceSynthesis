function [scene_dir,camera_dir, camerac_dir] = compute_CreateScene(folder_index,frame_index,p)
index = (folder_index-1)*p.frame_size + frame_index;
scene_index_start = (index-1)*p.view_size_query + 1;
scene_index_end = index*p.view_size_query;
scene_name = strcat('Scene_',int2str(scene_index_start),'_',int2str(scene_index_end));
scene_dir = fullfile(p.SCENE_DIR,scene_name);
mkdir(p.SCENE_DIR,scene_name);
scene_dir = strcat(scene_dir,'\');

camera_dir = fullfile(p.SCENEV_DIR,strcat(scene_name,'.mat'));

mkdir(p.CAMERAC_DIR,scene_name);
camerac_dir = fullfile(p.CAMERAC_DIR,scene_name);
camerac_dir = strcat(camerac_dir,'\');
end