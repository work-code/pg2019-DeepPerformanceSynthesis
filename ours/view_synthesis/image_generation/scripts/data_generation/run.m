dataSet = 'PVHM';
name = {'C1'};%{'C1';'C2';'C3';'C4';'C5';'C6';'C7'}
p = setting_up(dataSet,name);

%-----------------SpatialImage----------------
%Compute_SceneRendering(p);

%-------------------GQN Data-----------------------
%resolution = 128;
% Compute_SceneRendering_GQN_TrainTest(p,resolution);

% view_bias = 0;
% for g = 2:5
%     view_size = g; % 2 5
%     Compute_SceneRendering_GQN_Val(p,resolution,view_bias,view_size);
% end


%-------------------Super resolution------------
% low_resolution = 128;
% super_resolution = 512;
% Compute_SuperResolution_Data(p,low_resolution,super_resolution);



