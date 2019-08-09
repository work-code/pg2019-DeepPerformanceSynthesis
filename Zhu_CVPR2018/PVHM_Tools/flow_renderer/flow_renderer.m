%==========================================================================
% ===Introduction===
% This matlab script is for rendering backward flow, from the source image
% (front image, view_num = 9) to target images (view_num = 0:18 exclude 9).
% The default resolution is 200 x 200.  If the resolution is required to be
% changed, rewrite the intrinsic.txt accordingly.  The output backward 
% flows are stored as binary file.  The ply models will be produced as 
% intermediate files.
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

dataset_dir =  'H:\\data\\PVHM_0-9999\\'; % dataset directory
output_dir = sprintf('%sbackward_flow\\',dataset_dir); % output directory
view_num = 18; % view number
model_num = 10000; % model number


% add path to matlab
addpath('H:\data\PVHM_Tools\flow_renderer\functions')

% read intrinsic matrix and extrinsic matrix
K_array = load('intrinsic.txt');
Rt_array = load('extrinsic.txt');

% assign source intrinsic and extrinsic matrices
K_src = K_array(28:30,:);
Rt_src = Rt_array(28:30,:);

% make output folder if it doesn't exist
if ~exist(output_dir,'dir') 
    mkdir(output_dir);
end

% start the processing
% if you wanna enable multithreading to speed up the process, try replace 'for' with 'parfor'
for i_model = 1000:(model_num-1)
    
    test_msg = sprintf('processing %d / %d model',i_model, model_num-1);
    disp(test_msg)
    
    model_dir = sprintf('%s%d-%d\\',dataset_dir,floor(i_model/200)*200,floor(i_model/200)*200+199);
    ply_save_dir = sprintf('%s%d-%d\\ply\\',dataset_dir,floor(i_model/200)*200,floor(i_model/200)*200+199);
    if ~exist(ply_save_dir,'dir') 
        mkdir(ply_save_dir);
    end
    
    for i_view = 0:view_num
        % assign K and Rt
        K_tgt = K_array((i_view*3+1):(i_view*3+3),:);
        Rt_tgt = Rt_array((i_view*3+1):(i_view*3+3),:);
        
        % assign input file name
        obj_filename = sprintf('m_%d.obj',i_model-floor(i_model/200)*200);
        ply_filename = sprintf('m_%d.ply',i_model-floor(i_model/200)*200);
        
        % compute pose flow
        [FlowX, FlowY] = poseflow_krt(model_dir,ply_save_dir,obj_filename,ply_filename,K_src,Rt_src,K_tgt,Rt_tgt);
        
        % save the pose flow image
        filename_x = sprintf('%s%d_%d_u.bin',output_dir,i_model,i_view);
        filename_y = sprintf('%s%d_%d_v.bin',output_dir,i_model,i_view);
        
        % save files
        f_x = fopen(filename_x,'w');
        fwrite(f_x,FlowX,'double');
        fclose(f_x);
        f_y = fopen(filename_y,'w');
        fwrite(f_y,FlowY,'double');
        fclose(f_y);
        
    end
end

disp('==================================================');
disp('All Finished!!');
disp('==================================================');