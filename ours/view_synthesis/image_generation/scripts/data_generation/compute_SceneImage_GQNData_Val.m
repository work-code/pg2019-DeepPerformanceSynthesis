function compute_SceneImage_GQNData_Val(p,bias,view_bias,view_size)
list=dir(fullfile(p.GQNDATA_SCENE_DIR));
fileNum=size(list,1)-2; 
parfor g = 1:size(p.scene_name,1)
    folder_name = char(p.scene_name{g,1});
    index = strfind(folder_name,'_');
    startN = folder_name(1,index(1,1)+1:index(1,2)-1);
    startN = str2num(startN);
    endN = folder_name(1,index(1,2)+1:size(folder_name,2));
    endN = str2num(endN);
    original_scnen_dir = fullfile(p.SCENE_DIR,folder_name);
    for t = startN:endN
        scene_name = strcat('Scene',int2str(t-bias+fileNum));
        scene_dir = fullfile(p.GQNDATA_SCENE_DIR,scene_name);
        mkdir(p.GQNDATA_SCENE_DIR,scene_name);
        query_index = t - startN + 1;  
        query_index = query_index + p.view_size_observation;
        view_index = compute_SpatialData_SceneImage_Val_ViewIndex(p,view_bias,view_size);%1-20
        view_index = view_index + p.view_size_observation;
        compute_SpatialData_SceneImage(view_index,scene_dir,original_scnen_dir,query_index,p);
        
        orginal_camera_dir = fullfile(p.SCENEV_DIR,strcat(folder_name,'.mat'));
        view = loadFile(orginal_camera_dir);
        temp = view(view_index,:);
        temp2 = view(query_index,:);
        view_new = [temp;temp2];
        
        new_camera_dir = fullfile(p.GQNDATA_SCENEV_DIR,strcat(scene_name,'.mat'));
        saveFile(new_camera_dir,view_new)
    end
end
end


function view_index = compute_SpatialData_SceneImage_Val_ViewIndex(p,view_bias,view_size)
step = ceil(p.view_size_query/view_size);
view_index = zeros(view_size,1);
start = 1;
for g = 1:view_size
    view_index(g,1) = start + step*(g-1);
end
for g = 1:view_size
    view_index(g,1) = mod(view_bias+view_index(g,1),p.view_size_query);
end
end