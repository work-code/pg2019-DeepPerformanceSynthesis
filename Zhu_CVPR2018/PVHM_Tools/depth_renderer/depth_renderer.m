%==========================================================================
% ===Introduction===
% This matlab script is for rendering depth for each model at the front
% view (view_num = 9). The default resolution is 200 x 200.  If the 
% resolution is required to be changed, rewrite the intrinsic.txt 
% accordingly.  The output backward flows are stored as binary file.  The 
% ply models will be produced as intermediate files.
% ===Usage===
% (1) Uncompress the dataset to a folder and update the folder location to
%     variable 'dataset_dir'.
% (2) The default output directory is '[dataset_location]/backward_flow/'.
%     Rewrite it if needed.
% (3) Run the script.
% 
% Hao Zhu
%==========================================================================

clear all
clc

dataset_dir =  'H:\data\PVHM_0-9999\'; % dataset directory
output_dir = sprintf('%sdepth\\',dataset_dir); % output directory
model_num = 10000; % model number

% add path to matlab
addpath('functions')

% read K Rt
K_array = load('intrinsic.txt');
Rt_array = load('extrinsic.txt');

% assign K and Rt
K = K_array(28:30,:);
Rt = Rt_array(28:30,:);

% make output folder if it doesn't exist
if ~exist(output_dir,'dir') 
    mkdir(output_dir);
end

% start the iteration
% if you wanna enable multithreading to speed up the process, try replace 'for' with 'parfor'
for i_model = 999:(model_num-1)

    test_msg = sprintf('processing %d / %d model',i_model, model_num-1);
    disp(test_msg)
    
    model_dir = sprintf('%s%d-%d\\',dataset_dir,floor(i_model/200)*200,floor(i_model/200)*200+199);
    ply_save_dir = sprintf('%s%d-%d\\ply\\',dataset_dir,floor(i_model/200)*200,floor(i_model/200)*200+199);
    if ~exist(ply_save_dir,'dir') 
        mkdir(ply_save_dir);
    end
    
    % assign input file name
    obj_filename = sprintf('m_%d.obj',i_model-floor(i_model/200)*200);
    ply_filename = sprintf('m_%d.ply',i_model-floor(i_model/200)*200);

    % compute pose flow
    depth_map = DepthMap_front(model_dir,ply_save_dir,obj_filename,ply_filename,K,Rt);

    % save the pose flow image
    filename = sprintf('%s%d_9_d.bin',output_dir,i_model);

    f_d = fopen(filename,'w');
    fwrite(f_d,depth_map,'double');
    fclose(f_d);

end

disp('==================================================');
disp('All Finished!!');
disp('==================================================');
