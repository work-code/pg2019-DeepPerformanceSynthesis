function p = setting_upc(dataSet)
%----------------------------------------------------------Variable----------------------------------------------------------------------
p.folder_index = [1;2;3];

p.folder_train_index_original = [1;2];
p.folder_test_index_original = [3];
p.folder_val_index_original = [3];

p.folder_train_index = [1;2];
p.folder_test_index = [3];
p.folder_val_index = [3];

p.frame_size = 200;
p.resolution_x = 128;
p.resolution_y = 128;
p.resolution_x_hr = p.resolution_x*4;
p.resolution_y_hr = p.resolution_x*4;
p.view_size_observation = 4;
p.view_size_query = 20;

p.blender_location = '"D:\programSetup\Blender\blender.exe"';
p.single_renderer_location = 'G:\2-paper\ResearchWork5\ResearchWork5_zz_finalversion\code\ours\view_synthesis\image_generation\scripts\data_generation\single_model_renderer.py';
%-----------------------------------------------------------Root-------------------------------------------------------------------------

PROJECT_ROOT_DIR        = get_father_dic(pwd,2);
DATA_ROOT_DIR           = fullfile(PROJECT_ROOT_DIR, 'data', dataSet);
p.ROOT_DIR              = DATA_ROOT_DIR;
p.MODEL_DIR             = fullfile(DATA_ROOT_DIR,'mesh');
mkdir(DATA_ROOT_DIR,'mesh');
p.OBJ_DIR             = fullfile(p.MODEL_DIR,'obj');
mkdir(p.MODEL_DIR,'obj');
p.RENDER_DIR             = fullfile(DATA_ROOT_DIR,'rendering');
mkdir(DATA_ROOT_DIR,'rendering');
p.GQNDATA_DIR             = fullfile(DATA_ROOT_DIR,'gqndata');
mkdir(DATA_ROOT_DIR,'gqndata');
p.SUPERDATA_SIR           = fullfile(DATA_ROOT_DIR,'superdata');
mkdir(DATA_ROOT_DIR,'superdata');
p.DEMONET_DIR           = fullfile(DATA_ROOT_DIR,'demonetwork');
mkdir(DATA_ROOT_DIR,'demonetwork');
p.EVAL_DIR           = fullfile(DATA_ROOT_DIR,'eval');
mkdir(DATA_ROOT_DIR,'eval');
p.MSE_DIR            = fullfile(DATA_ROOT_DIR,'mse');
mkdir(DATA_ROOT_DIR,'mse');
p.SSIM_DIR            = fullfile(DATA_ROOT_DIR,'ssim');
mkdir(DATA_ROOT_DIR,'ssim');

SCRIPTS_ROOT_DIR        = fullfile(PROJECT_ROOT_DIR,'scripts');
CODE_ROOT_DIR           = fullfile(SCRIPTS_ROOT_DIR, 'common');
MESH_ROOT_DIR           = fullfile(CODE_ROOT_DIR, '/meshCode/');


addpath(CODE_ROOT_DIR);
addpath(MESH_ROOT_DIR);
addpath(pwd);
end
