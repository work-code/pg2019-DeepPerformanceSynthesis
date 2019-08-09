function compute_SceneImage(p,flag)

parfor t = 1:size(p.folder_name,1)
    folder_name = char(p.folder_name{t,1});
    if flag == 0
        camera_bias = rand(1)*pi*2;
    end
    for g = 1:p.frame_size
        model_name = strcat('m_',int2str(g-1),'.obj');
        c_name = folder_name(1,1:2);
        model_location = fullfile(p.OBJ_DIR,c_name,folder_name,model_name);
        [scene_dir,camera_dir,camerac_dir] = compute_CreateScene(t,g,p);
        
        if flag == 1
            fileName = replace(camerac_dir,strcat('\',int2str(p.cur_resolution),'\'),strcat('\',int2str(128),'\'));
            fileName = fullfile(fileName,'camera_bias.mat');
            camera_bias = loadFile(fileName);
        end
        
        command = sprintf('%s --background --python %s -- %d %d %f %d %d %s %s %s',...
            p.blender_location, p.single_renderer_location,...
            p.view_size_observation, p.view_size_query, camera_bias,... 
            p.cur_resolution, p.cur_resolution, model_location, scene_dir, camerac_dir);
        compute_RenderingImage(command,strcat(folder_name,model_name),scene_dir);
        
        compute_CameraPosition(camera_dir,camerac_dir,camera_bias);
    end
    
    
end
end

