function compute_SpatialData_SceneImage(view_index,scene_dir,original_scnen_dir,query_index,p)

for g = 1:size(view_index,1)
    image_dir = fullfile(original_scnen_dir,strcat(int2str(view_index(g,1)),'.jpg'));
    image = imread(image_dir);
    imwrite(image,fullfile(scene_dir,strcat(int2str(g),'.jpg')),'quality',100);
end
image_dir = fullfile(original_scnen_dir,strcat(int2str(query_index),'.jpg'));
S = imread(image_dir);
imwrite(S,fullfile(scene_dir,strcat(int2str(size(view_index,1)+1),'.jpg')),'quality',100);

image_dir = strrep(image_dir,strcat('\',int2str(p.resolution_x),'\'),strcat('\',int2str(p.resolution_x*4),'\'));
S = imread(image_dir);
imwrite(S,fullfile(scene_dir,'super.jpg'),'quality',100);

end