function computer_SuperResolution_DataImage(p,low_resolution,super_resolution)
scene_dir = fullfile(p.RENDER_DIR,int2str(low_resolution),'scene');
super_scene_dir = fullfile(p.RENDER_DIR,int2str(super_resolution),'scene');
for g = 1:size(p.folder_name,1)
    folder_name = char(p.folder_name{g,1});
    scene_path = fullfile(scene_dir,folder_name);
    super_scene_path = fullfile(super_scene_dir,folder_name);
    
    scene_name = strcat('Scene',int2str(g));
    mkdir(p.SUPERDATA_SCENE_SIR,scene_name);
    scene_path_sr = fullfile(p.SUPERDATA_SCENE_SIR,scene_name);
    
    mkdir(p.SUPERDATA_SUPERSCENE_SIR,scene_name);
    super_scene_path_sr = fullfile(p.SUPERDATA_SUPERSCENE_SIR,scene_name);
    
    for m = 1:p.view_size_observation + p.view_size_query
        image_path = fullfile(scene_path,strcat(int2str(m),'.jpg'));
        copyfile(image_path,scene_path_sr);
        
        image_path = fullfile(super_scene_path,strcat(int2str(m),'.jpg')); 
        copyfile(image_path,super_scene_path_sr);
        
    end
end
end