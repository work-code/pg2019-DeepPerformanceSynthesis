function compute_CameraPosition_Val(camera_dir,camerac_dir,frame,flag)
file_location = fullfile(camerac_dir,'camera_location.txt');
camera_location = textread(file_location);
camera_location = camera_location(:,1:3);

file_location = fullfile(camerac_dir,'camera_rotation.txt');
camera_rotation = textread(file_location);
camera_rotation = camera_rotation(:,1:3);

yaw = repmat(0,size(camera_rotation,1),1);
pitch = repmat(0,size(camera_rotation,1),1);
frame = repmat(frame,size(camera_rotation,1),1);
flag = repmat(flag,size(camera_rotation,1),1);

camera_position = [camera_location,yaw,pitch,frame,flag];

saveFile(camera_dir,camera_position);
end