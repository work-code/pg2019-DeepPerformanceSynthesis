function [scene_dir,camera_dir,camerac_dir] = compute_CreateScene_Val(folder_index,model_index,view_index,p)
scene_index = (folder_index-1)*p.view_size_val*p.frame_size + (model_index-1)*p.view_size_val + view_index;
scene_dir = fullfile(p.SCENE_DIR,strcat('Scene',int2str(scene_index)));
mkdir(p.SCENE_DIR,strcat('Scene',int2str(scene_index)));
scene_dir = strcat(scene_dir,'\');

camera_dir = fullfile(p.SCENEV_DIR,strcat('Scene',int2str(scene_index),'.mat'));

mkdir(p.CAMERAC_DIR,strcat('Scene',int2str(scene_index)));
camerac_dir = fullfile(p.CAMERAC_DIR,strcat('Scene',int2str(scene_index)));
camerac_dir = strcat(camerac_dir,'\');
end