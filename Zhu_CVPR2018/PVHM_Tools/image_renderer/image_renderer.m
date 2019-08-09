%==========================================================================
% ===Introduction===
% This matlab script is for rendering images from textured obj mesh models 
% in pose-varying human dataset.  The software 'Blender' is required for 
% rendering.  'Blender' is an open-source software and could be freely
% downloaded from https://www.blender.org/.  We recommend 2.79a version 
% and any versions older than 2.79 will NOT work.  Note that we found 
% in some machine the batch-processing blender may fail in a few models,
% so in this script after the whole rendering process, the rendered list 
% will be checked and the missing images will be re-rendered.  If you wanna
% skip the checking and re-rendering phase, set 'missing_check = false' in
% line 38.  The 4th channel of the output image is the foreground mask.
% ===Usage===
% (1) Update the blender location to variable % 'blender_location'.
% (2) Uncompress the dataset to a folder and update the folder location to
%     variable 'dataset_dir'.
% (3) Other options: output_dir (default is './rendered_images/')
%                    resolution_x/y (default is 200x200)
%     More rendering options could be found in 'single_model_renderer.py'.
% (4) Run the script.
% 
% Hao Zhu
%==========================================================================

clear all
clc

% please update these parameteres before run
blender_location = '"C:\Program Files\Blender Foundation\blender-2.80-3d8cbb534f82-win64\blender.exe"'; % change it to the blender setup directory in your machine
dataset_dir = 'H:\data\PVHM_0-9999\'; % change it to the directory where the model data are uncompressed
output_dir = sprintf('%srendered_images_1\\',dataset_dir); % this is the default output directory, change it if you hope to output to other location

% static parameters
single_renderer_location = './single_model_renderer.py';
resolution_x = 200;
resolution_y = 200;
model_num = 10000;
missing_check = true;

% make the new file
if ~exist(output_dir,'dir') 
    mkdir(output_dir)
end

% clear the rendered_list_tmp file
if missing_check == true
    if exist('./rendered_list_tmp.txt','file') 
        delete('./rendered_list_tmp.txt')
    end
end

% rendering loop
for num = 0:model_num-1
    model_location = sprintf('%s%d-%d\\m_%d.obj',dataset_dir,floor(num/200)*200,floor(num/200)*200+199,num-floor(num/200)*200);
    command = sprintf('%s --background --python %s -- %d %d %d %s %s',blender_location, single_renderer_location, num, resolution_x, resolution_y, model_location, output_dir);
    system(command);
    disp('==================================================')
    info = sprintf('model %d rendered.',num);
    disp(info);
    disp('==================================================')
end

if missing_check == true
    % check the rendered_list_tmp and make missing_list
    rendered_list = load('./rendered_list_tmp.txt');

    missing_list = zeros(model_num,1);
    for num = 1:size(rendered_list,1)
        missing_list(rendered_list(num)+1) = 1;
    end

    % re-rendered if need
    recycle_num = 0;
    while(sum(missing_list) ~= model_num)
        recycle_num = recycle_num + 1;
        if recycle_num > 20
            disp('==================================================');
            disp('Interrupted. Not all the images are rendered due to');
            disp('some problems, please check rendered_list_tmp.txt to');
            disp('see which model failed.');
            disp('==================================================');
            return;
        end
        for num_t = 1:model_num
            if missing_list(num_t) == 0
                    num = num_t-1;
                    model_location = sprintf('%s%d-%d\\m_%d.obj',dataset_dir,floor(num/200)*200,floor(num/200)*200+199,num-floor(num/200)*200);
                    command = sprintf('%s --background --python %s -- %d %d %d %s %s',blender_location, single_renderer_location, num, resolution_x, resolution_y, model_location, output_dir);
                    system(command);
            end
        end
    end
end
disp('==================================================');
disp('All Finished!!');
disp('==================================================');