function compute_CameraPosition(camera_dir,camerac_dir,camera_bias)
file_location = fullfile(camerac_dir,'camera_location.txt');
camera_location = textread(file_location);
camera_location = camera_location(:,1:3);

file_location = fullfile(camerac_dir,'camera_rotation.txt');
camera_rotation = textread(file_location);
camera_rotation = camera_rotation(:,1:3);

yaw = repmat(1,size(camera_rotation,1),1);
pitch = repmat(1,size(camera_rotation,1),1);

camera_position = [camera_location,yaw,pitch];

saveFile(camera_dir,camera_position);

camera_bias_dir = fullfile(camerac_dir,'camera_bias.mat');
saveFile(camera_bias_dir,camera_bias);
end