function Compute_SceneRendering_GQN_TrainTest(p,resolution)
temp_path = p.GQNDATA_DIR; 
temp_path2 = p.RENDER_DIR;

p.scene_name = p.folder_train; p = compute_SetRoot(p,int2str(resolution),'train');
compute_SceneImage_GQNData(p);
p.GQNDATA_DIR = temp_path;
p.RENDER_DIR = temp_path2;

end

function p = compute_SetRoot(p,resolution,type)
mkdir(p.GQNDATA_DIR,resolution);
p.GQNDATA_DIR = fullfile(p.GQNDATA_DIR,resolution);
mkdir(p.GQNDATA_DIR,type);
p.GQNDATA_DIR = fullfile(p.GQNDATA_DIR,type);
mkdir(p.GQNDATA_DIR,'scene');
p.GQNDATA_SCENE_DIR = fullfile(p.GQNDATA_DIR,'scene');
mkdir(p.GQNDATA_DIR,'sceneview');
p.GQNDATA_SCENEV_DIR = fullfile(p.GQNDATA_DIR,'sceneview');

p.RENDER_DIR = fullfile(p.RENDER_DIR,resolution);
p.SCENE_DIR = fullfile(p.RENDER_DIR,'scene');
p.SCENEV_DIR = fullfile(p.RENDER_DIR,'sceneview');
p.CAMERAC_DIR = fullfile(p.RENDER_DIR,'cameracalibration');
end

