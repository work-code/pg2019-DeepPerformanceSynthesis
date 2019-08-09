function Compute_SceneRendering(p)
temp_path = p.RENDER_DIR; 
p.folder_name = p.folder; 

%128
flag = 0; p.cur_resolution = 128; p = compute_SetRoot(p,p.cur_resolution);
compute_SceneImage(p,flag);
p.RENDER_DIR = temp_path;

%512
flag = 1; p.cur_resolution = 128*4; p = compute_SetRoot(p,p.cur_resolution);
compute_SceneImage(p,flag);
p.RENDER_DIR = temp_path;

end

function p = compute_SetRoot(p,resolution)
mkdir(p.RENDER_DIR,int2str(resolution));
p.RENDER_DIR = fullfile(p.RENDER_DIR,int2str(resolution));
mkdir(p.RENDER_DIR,'scene');
p.SCENE_DIR = fullfile(p.RENDER_DIR,'scene');
mkdir(p.RENDER_DIR,'sceneview');
p.SCENEV_DIR = fullfile(p.RENDER_DIR,'sceneview');
mkdir(p.RENDER_DIR,'cameracalibration');
p.CAMERAC_DIR = fullfile(p.RENDER_DIR,'cameracalibration');
end
