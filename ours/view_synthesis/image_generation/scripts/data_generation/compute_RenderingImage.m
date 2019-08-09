function compute_RenderingImage(command,model_name,scene_dir)
system(command);
disp('==================================================')
info = sprintf('model %s rendered.',model_name);
disp(info);
disp('==================================================')

dirOutput = dir(fullfile(scene_dir,'*.png'));
image_name ={dirOutput.name}';
for g = 1:size(image_name,1)
    image = char(image_name{g,1}); 
    image = image(1,1:strfind(image,'.png')-1);
    S = imread(fullfile(scene_dir,strcat(image,'.png')));
    imwrite(S,fullfile(scene_dir,strcat(image,'.jpg')),'quality',100);
    %delete(fullfile(scene_dir,strcat(image,'.png')));
end

end