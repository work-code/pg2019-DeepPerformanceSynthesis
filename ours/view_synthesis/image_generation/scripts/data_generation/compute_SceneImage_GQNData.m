function compute_SceneImage_GQNData(p)
parfor g = 1:size(p.scene_name,1)
    folder_name = char(p.scene_name{g,1});
   
    original_scnen_dir = fullfile(p.SCENE_DIR,folder_name);
    
    scene_name = strcat('Scene',int2str(g));
    scene_dir = fullfile(p.GQNDATA_SCENE_DIR,scene_name);
    mkdir(p.GQNDATA_SCENE_DIR,scene_name);
    for t = 1:p.view_size_query  
        query_index = t + p.view_size_observation;
        image_dir = fullfile(original_scnen_dir,strcat(int2str(query_index),'.jpg'));
        S = imread(image_dir);
        imwrite(S,fullfile(scene_dir,strcat(int2str(t),'.jpg')),'quality',100);
    end
    orginal_camera_dir = fullfile(p.SCENEV_DIR,strcat(folder_name,'.mat'));
    view = loadFile(orginal_camera_dir);
    view_new = view(p.view_size_observation+1:p.view_size_observation+p.view_size_query,:);
        
    new_camera_dir = fullfile(p.GQNDATA_SCENEV_DIR,strcat(scene_name,'.mat'));
    saveFile(new_camera_dir,view_new);
end
end