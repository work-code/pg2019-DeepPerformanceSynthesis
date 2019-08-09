function Compute_SceneRendering_GQN_Val(p,resolution,view_bias,view_size)
temp_path = p.GQNDATA_DIR; 
temp_path2 = p.RENDER_DIR;

bias = (p.folder_train_index(1,1)-1)*p.frame_size*p.view_size_query;
p.scene_name = p.folder_train; p = compute_SetRoot(p,int2str(resolution),'val',view_bias,view_size);
compute_SceneImage_GQNData_Val(p,bias,view_bias,view_size);
p.GQNDATA_DIR = temp_path;
p.RENDER_DIR = temp_path2;

bias = (p.folder_test_index(1,1)-1)*p.frame_size*p.view_size_query;
p.scene_name = p.folder_test; p = compute_SetRoot(p,int2str(resolution),'val',view_bias,view_size);
compute_SceneImage_GQNData_Val(p,bias,view_bias,view_size);
p.GQNDATA_DIR = temp_path;
p.RENDER_DIR = temp_path2;

end


function p = compute_SetRoot(p,resolution,type,view_bias,view_size)
mkdir(p.GQNDATA_DIR,resolution);
p.GQNDATA_DIR = fullfile(p.GQNDATA_DIR,resolution);
mkdir(p.GQNDATA_DIR,type);
p.GQNDATA_DIR = fullfile(p.GQNDATA_DIR,type);
mkdir(p.GQNDATA_DIR,strcat('bias_',int2str(view_bias)));
p.GQNDATA_DIR = fullfile(p.GQNDATA_DIR,strcat('bias_',int2str(view_bias)));
mkdir(p.GQNDATA_DIR,strcat('observation_',int2str(view_size)));
p.GQNDATA_DIR = fullfile(p.GQNDATA_DIR,strcat('observation_',int2str(view_size)));
mkdir(p.GQNDATA_DIR,'scene');
p.GQNDATA_SCENE_DIR = fullfile(p.GQNDATA_DIR,'scene');
mkdir(p.GQNDATA_DIR,'sceneview');
p.GQNDATA_SCENEV_DIR = fullfile(p.GQNDATA_DIR,'sceneview');

p.RENDER_DIR = fullfile(p.RENDER_DIR,resolution);
p.SCENE_DIR = fullfile(p.RENDER_DIR,'scene');
p.SCENEV_DIR = fullfile(p.RENDER_DIR,'sceneview');
p.CAMERAC_DIR = fullfile(p.RENDER_DIR,'cameracalibration');
end

